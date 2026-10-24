#pragma once

// The link between two stages of a model whose layers are split across
// processes (a layer slice each, QwenTargetDimensions::sliced): the stage
// that holds the model's last layers drives the one before it over one TCP
// connection on loopback, which whatever carries it between machines may
// relay.
//
// Frames, both ways: a little-endian u32 payload length, then the payload,
// whose first byte is its kind: 1 for raw bytes, 2 for a tensor (a
// little-endian u32 length of its metadata, the metadata, then f32 values).
// A carrier may convert a tensor's values in flight, so a tensor arrives as
// f32 whatever precision it travelled in; raw bytes arrive as sent.
//
// Every request is one frame and has one answer frame, in order: the node
// that relays the link between machines posts each frame and writes back
// its answer (engine/StageServant.hpp holds the requests).

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace splash::engine {

class StageLinkError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

inline constexpr uint8_t kStageFrameRaw = 1;
inline constexpr uint8_t kStageFrameTensor = 2;
// The largest payload either side accepts.
inline constexpr uint32_t kStageMaxFrame = 1U << 30;

// One connected end: blocking, framed, closed with the object.
class StageSocket final {
public:
  StageSocket() = default;
  explicit StageSocket(int descriptor) noexcept : fd_(descriptor) {}
  StageSocket(StageSocket &&other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  StageSocket &operator=(StageSocket &&other) noexcept;
  StageSocket(const StageSocket &) = delete;
  StageSocket &operator=(const StageSocket &) = delete;
  ~StageSocket();

  [[nodiscard]] bool open() const noexcept { return fd_ >= 0; }
  [[nodiscard]] int descriptor() const noexcept { return fd_; }
  void sendRaw(std::span<const uint8_t> bytes);
  // A tensor of `values` with metadata `meta` (UTF-8, usually JSON).
  void sendTensor(std::string_view meta, std::span<const float> values);
  // The next frame's kind and payload past the kind byte; throws at EOF.
  uint8_t receive(std::vector<uint8_t> &payload);
  [[nodiscard]] std::vector<uint8_t> receiveRaw();
  // A tensor frame's values, its metadata in `meta`.
  [[nodiscard]] std::vector<float> receiveTensor(std::string &meta);

private:
  void writeAll(const void *data, size_t bytes);
  void readAll(void *data, size_t bytes);
  void sendFrame(uint8_t kind, std::span<const uint8_t> head, std::span<const uint8_t> body);

  int fd_ = -1;
};

// Listens on 127.0.0.1:port; accept() takes the next connection.
class StageListener final {
public:
  explicit StageListener(uint16_t port);
  [[nodiscard]] StageSocket accept();

private:
  StageSocket socket_;
};
// Connects to 127.0.0.1:port.
[[nodiscard]] StageSocket connectStage(uint16_t port);

// A little-endian encoder and decoder of a raw frame's fields.
class StageWriter final {
public:
  void u8(uint8_t value) { bytes_.push_back(value); }
  void u32(uint32_t value);
  void u64(uint64_t value);
  void words(std::span<const uint32_t> values);
  void text(std::string_view value);
  [[nodiscard]] const std::vector<uint8_t> &bytes() const noexcept { return bytes_; }

private:
  std::vector<uint8_t> bytes_;
};

class StageReader final {
public:
  explicit StageReader(std::span<const uint8_t> bytes) noexcept : bytes_(bytes) {}
  [[nodiscard]] uint8_t u8();
  [[nodiscard]] uint32_t u32();
  [[nodiscard]] uint64_t u64();
  [[nodiscard]] std::vector<uint32_t> words();
  [[nodiscard]] std::string text();
  [[nodiscard]] bool done() const noexcept { return at_ == bytes_.size(); }

private:
  void need(size_t bytes) const;
  std::span<const uint8_t> bytes_;
  size_t at_ = 0;
};

} // namespace splash::engine
