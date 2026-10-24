#include "engine/StageLink.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <bit>
#include <stdexcept>
#include <utility>

namespace splash::engine {
namespace {

[[noreturn]] void failErrno(std::string_view what) {
  throw StageLinkError(std::string(what) + ": " + std::strerror(errno));
}

void putU32(uint8_t *out, uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) out[byte] = static_cast<uint8_t>(value >> (8 * byte));
}

uint32_t getU32(const uint8_t *in) {
  uint32_t value = 0;
  for (int byte = 0; byte < 4; ++byte) value |= uint32_t{in[byte]} << (8 * byte);
  return value;
}

sockaddr_in loopback(uint16_t port) {
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return address;
}

// Requests and answers are small and wait for each other: no Nagle delay.
void noDelay(int fd) {
  const int on = 1;
  static_cast<void>(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on));
#ifdef SO_NOSIGPIPE
  static_cast<void>(setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on));
#endif
}

} // namespace

StageSocket &StageSocket::operator=(StageSocket &&other) noexcept {
  if (this != &other) {
    if (fd_ >= 0) ::close(fd_);
    fd_ = std::exchange(other.fd_, -1);
  }
  return *this;
}

StageSocket::~StageSocket() {
  if (fd_ >= 0) ::close(fd_);
}

void StageSocket::writeAll(const void *data, size_t bytes) {
  const auto *at = static_cast<const uint8_t *>(data);
  while (bytes) {
    const ssize_t written = ::send(fd_, at, bytes, 0);
    if (written < 0) {
      if (errno == EINTR) continue;
      failErrno("stage link write");
    }
    at += written;
    bytes -= static_cast<size_t>(written);
  }
}

void StageSocket::readAll(void *data, size_t bytes) {
  auto *at = static_cast<uint8_t *>(data);
  while (bytes) {
    const ssize_t read = ::recv(fd_, at, bytes, 0);
    if (read < 0) {
      if (errno == EINTR) continue;
      failErrno("stage link read");
    }
    if (read == 0) throw StageLinkError("the stage link closed");
    at += read;
    bytes -= static_cast<size_t>(read);
  }
}

void StageSocket::sendFrame(uint8_t kind, std::span<const uint8_t> head, std::span<const uint8_t> body) {
  if (fd_ < 0) throw StageLinkError("the stage link is not open");
  const uint64_t length = 1 + uint64_t{head.size()} + body.size();
  if (length > kStageMaxFrame) throw StageLinkError("a stage frame is too large");
  std::array<uint8_t, 5> prefix{};
  putU32(prefix.data(), static_cast<uint32_t>(length));
  prefix[4] = kind;
  writeAll(prefix.data(), prefix.size());
  if (!head.empty()) writeAll(head.data(), head.size());
  if (!body.empty()) writeAll(body.data(), body.size());
}

void StageSocket::sendRaw(std::span<const uint8_t> bytes) { sendFrame(kStageFrameRaw, {}, bytes); }

void StageSocket::sendTensor(std::string_view meta, std::span<const float> values) {
  std::vector<uint8_t> head(4 + meta.size());
  putU32(head.data(), static_cast<uint32_t>(meta.size()));
  std::memcpy(head.data() + 4, meta.data(), meta.size());
  static_assert(std::endian::native == std::endian::little, "the stage link writes f32 little-endian");
  sendFrame(kStageFrameTensor, head,
            {reinterpret_cast<const uint8_t *>(values.data()), values.size_bytes()});
}

uint8_t StageSocket::receive(std::vector<uint8_t> &payload) {
  if (fd_ < 0) throw StageLinkError("the stage link is not open");
  std::array<uint8_t, 4> prefix{};
  readAll(prefix.data(), prefix.size());
  const uint32_t length = getU32(prefix.data());
  if (!length || length > kStageMaxFrame) throw StageLinkError("a stage frame has an invalid length");
  uint8_t kind = 0;
  readAll(&kind, 1);
  payload.resize(length - 1);
  if (!payload.empty()) readAll(payload.data(), payload.size());
  return kind;
}

std::vector<uint8_t> StageSocket::receiveRaw() {
  std::vector<uint8_t> payload;
  if (receive(payload) != kStageFrameRaw) throw StageLinkError("expected a raw stage frame");
  return payload;
}

std::vector<float> StageSocket::receiveTensor(std::string &meta) {
  std::vector<uint8_t> payload;
  if (receive(payload) != kStageFrameTensor) throw StageLinkError("expected a tensor stage frame");
  if (payload.size() < 4) throw StageLinkError("a tensor stage frame is truncated");
  const uint32_t metaBytes = getU32(payload.data());
  if (metaBytes > payload.size() - 4 || (payload.size() - 4 - metaBytes) % sizeof(float))
    throw StageLinkError("a tensor stage frame has an invalid layout");
  meta.assign(reinterpret_cast<const char *>(payload.data() + 4), metaBytes);
  std::vector<float> values((payload.size() - 4 - metaBytes) / sizeof(float));
  std::memcpy(values.data(), payload.data() + 4 + metaBytes, values.size() * sizeof(float));
  return values;
}

StageListener::StageListener(uint16_t port) {
  const int listener = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listener < 0) failErrno("stage link socket");
  socket_ = StageSocket(listener);
  const int on = 1;
  static_cast<void>(setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on));
  const sockaddr_in address = loopback(port);
  if (::bind(listener, reinterpret_cast<const sockaddr *>(&address), sizeof address) < 0)
    failErrno("stage link bind 127.0.0.1:" + std::to_string(port));
  if (::listen(listener, 16) < 0) failErrno("stage link listen");
}

StageSocket StageListener::accept() {
  for (;;) {
    const int fd = ::accept(socket_.descriptor(), nullptr, nullptr);
    if (fd < 0) {
      if (errno == EINTR) continue;
      failErrno("stage link accept");
    }
    noDelay(fd);
    return StageSocket(fd);
  }
}

StageSocket connectStage(uint16_t port) {
  const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) failErrno("stage link socket");
  StageSocket socket(fd);
  const sockaddr_in address = loopback(port);
  if (::connect(fd, reinterpret_cast<const sockaddr *>(&address), sizeof address) < 0)
    failErrno("stage link connect 127.0.0.1:" + std::to_string(port));
  noDelay(fd);
  return socket;
}

void StageWriter::u32(uint32_t value) {
  for (int byte = 0; byte < 4; ++byte) bytes_.push_back(static_cast<uint8_t>(value >> (8 * byte)));
}

void StageWriter::u64(uint64_t value) {
  for (int byte = 0; byte < 8; ++byte) bytes_.push_back(static_cast<uint8_t>(value >> (8 * byte)));
}

void StageWriter::words(std::span<const uint32_t> values) {
  u32(static_cast<uint32_t>(values.size()));
  for (const uint32_t value : values) u32(value);
}

void StageWriter::text(std::string_view value) {
  u32(static_cast<uint32_t>(value.size()));
  bytes_.insert(bytes_.end(), value.begin(), value.end());
}

void StageReader::need(size_t bytes) const {
  if (bytes > bytes_.size() - at_) throw StageLinkError("a stage frame is truncated");
}

uint8_t StageReader::u8() {
  need(1);
  return bytes_[at_++];
}

uint32_t StageReader::u32() {
  need(4);
  const uint32_t value = getU32(bytes_.data() + at_);
  at_ += 4;
  return value;
}

uint64_t StageReader::u64() {
  const uint64_t low = u32();
  return low | uint64_t{u32()} << 32;
}

std::vector<uint32_t> StageReader::words() {
  const uint32_t count = u32();
  need(uint64_t{count} * 4);
  std::vector<uint32_t> values(count);
  for (uint32_t &value : values) value = u32();
  return values;
}

std::string StageReader::text() {
  const uint32_t count = u32();
  need(count);
  std::string value(reinterpret_cast<const char *>(bytes_.data() + at_), count);
  at_ += count;
  return value;
}

} // namespace splash::engine
