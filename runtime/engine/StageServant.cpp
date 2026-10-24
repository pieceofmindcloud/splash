#include "engine/StageServant.hpp"

#include "engine/Cache.hpp"
#include "model/Runtime.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace splash::engine {
namespace {

constexpr uint8_t kDone = 0;
constexpr uint8_t kFailed = 1;

std::vector<float> joined(const model::StageRows &rows) {
  std::vector<float> values;
  values.reserve(rows.hidden.size() + rows.captured.size());
  values.insert(values.end(), rows.hidden.begin(), rows.hidden.end());
  values.insert(values.end(), rows.captured.begin(), rows.captured.end());
  return values;
}

void answerFailure(StageSocket &socket, std::string_view reason) {
  StageWriter answer;
  answer.u8(kFailed);
  answer.text(reason);
  socket.sendRaw(answer.bytes());
}

} // namespace

StageClient::StageClient(StageSocket socket, std::string_view secret, uint32_t firstLayer)
    : socket_(std::move(socket)) {
  StageWriter hello;
  hello.u8(kStageHello);
  hello.text(secret);
  hello.u32(firstLayer);
  const std::vector<uint8_t> greeting = call(hello);
  StageReader answer(greeting);
  const uint32_t first = answer.u32(), end = answer.u32();
  if (end != firstLayer)
    throw StageLinkError("the previous stage runs layers " + std::to_string(first) + " to " +
                         std::to_string(end) + ", which do not end where this stage starts (" +
                         std::to_string(firstLayer) + ")");
}

std::vector<uint8_t> StageClient::call(const StageWriter &request) {
  socket_.sendRaw(request.bytes());
  std::vector<uint8_t> payload;
  if (socket_.receive(payload) != kStageFrameRaw) throw StageLinkError("the previous stage answered rows");
  StageReader answer(payload);
  if (answer.u8() != kDone) throw std::runtime_error("the previous stage refused: " + answer.text());
  return {payload.begin() + 1, payload.end()};
}

model::StageRows StageClient::rows(const StageWriter &request) {
  socket_.sendRaw(request.bytes());
  std::vector<uint8_t> payload;
  if (socket_.receive(payload) == kStageFrameRaw) {
    StageReader answer(payload);
    if (answer.u8() == kDone) throw StageLinkError("the previous stage answered no rows");
    throw std::runtime_error("the previous stage refused: " + answer.text());
  }
  if (payload.size() < 4) throw StageLinkError("a stage rows tensor is truncated");
  const uint32_t metaBytes = uint32_t{payload[0]} | uint32_t{payload[1]} << 8 | uint32_t{payload[2]} << 16 |
                             uint32_t{payload[3]} << 24;
  if (metaBytes > payload.size() - 4) throw StageLinkError("a stage rows tensor is truncated");
  const std::string meta(reinterpret_cast<const char *>(payload.data() + 4), metaBytes);
  unsigned long long hiddenRows = 0, hiddenWidth = 0, capturedRows = 0, capturedWidth = 0;
  const uint64_t floats = (payload.size() - 4 - metaBytes) / sizeof(float);
  if (std::sscanf(meta.c_str(), "hidden=%llux%llu captured=%llux%llu", &hiddenRows, &hiddenWidth, &capturedRows,
                  &capturedWidth) != 4 ||
      (payload.size() - 4 - metaBytes) % sizeof(float) || floats != hiddenRows * hiddenWidth + capturedRows * capturedWidth)
    throw StageLinkError("a stage rows tensor does not match its row counts");
  model::StageRows result;
  result.hidden.resize(hiddenRows * hiddenWidth);
  result.captured.resize(capturedRows * capturedWidth);
  const uint8_t *values = payload.data() + 4 + metaBytes;
  std::memcpy(result.hidden.data(), values, result.hidden.size() * sizeof(float));
  std::memcpy(result.captured.data(), values + result.hidden.size() * sizeof(float),
              result.captured.size() * sizeof(float));
  return result;
}

model::StageRows StageClient::prefill(uint64_t requestId, uint64_t position, uint32_t count,
                                      std::span<const uint32_t> history,
                                      std::span<const model::StageCaptureSpan> captures) {
  if (history.size() < position + count) throw std::invalid_argument("a stage prefill's history is too short");
  StageWriter request;
  request.u8(kStagePrefill);
  request.u64(requestId);
  request.u64(position);
  request.u32(count);
  request.words(history.first(position + count));
  request.u32(static_cast<uint32_t>(captures.size()));
  for (const model::StageCaptureSpan &span : captures) {
    request.u32(span.begin);
    request.u32(span.end);
  }
  const std::lock_guard lock(mutex_);
  return rows(request);
}

model::StageRows StageClient::verify(std::span<const model::StageVerifyLane> lanes) {
  StageWriter request;
  request.u8(kStageVerify);
  request.u32(static_cast<uint32_t>(lanes.size()));
  for (const model::StageVerifyLane &lane : lanes) {
    request.u64(lane.requestId);
    request.u64(lane.position);
    request.u32(lane.retainedBefore);
    request.u32(lane.verifyRows);
    request.words(lane.tokens);
  }
  const std::lock_guard lock(mutex_);
  return rows(request);
}

void StageClient::end(uint64_t requestId, uint32_t retainedBefore) noexcept {
  try {
    StageWriter request;
    request.u8(kStageEnd);
    request.u64(requestId);
    request.u32(retainedBefore);
    const std::lock_guard lock(mutex_);
    static_cast<void>(call(request));
  } catch (const std::exception &) {
    // A previous stage that went away fails the next request that needs it.
  }
}

void StageServant::serve(StageSocket &socket) {
  std::unordered_set<uint64_t> opened;
  bool greeted = false;
  for (;;) {
    std::vector<uint8_t> frame;
    try {
      frame = socket.receiveRaw();
    } catch (const StageLinkError &) {
      break;
    }
    StageReader request(frame);
    try {
      const uint8_t op = request.u8();
      if (!greeted) {
        if (op != kStageHello || request.text() != secret_ || request.u32() != endLayer_) {
          answerFailure(socket, "this stage link refuses the connection");
          break;
        }
        greeted = true;
        StageWriter answer;
        answer.u8(kDone);
        answer.u32(firstLayer_);
        answer.u32(endLayer_);
        socket.sendRaw(answer.bytes());
        continue;
      }
      model::StageRows rows;
      uint64_t rowCount = 0, capturedCount = 0;
      {
        const std::lock_guard lock(mutex_);
        opened_ = &opened;
        if (op == kStagePrefill) {
          rows = prefill(request);
        } else if (op == kStageVerify) {
          rows = verify(request);
        } else if (op == kStageEnd) {
          end(request);
        } else {
          throw std::invalid_argument("unknown stage request " + std::to_string(op));
        }
        rowCount = rowsOf_;
        capturedCount = capturedOf_;
        opened_ = nullptr;
      }
      if (op == kStageEnd) {
        StageWriter answer;
        answer.u8(kDone);
        socket.sendRaw(answer.bytes());
        continue;
      }
      const uint64_t hiddenWidth = rowCount ? rows.hidden.size() / rowCount : 0;
      const uint64_t capturedWidth = capturedCount ? rows.captured.size() / capturedCount : 0;
      socket.sendTensor("hidden=" + std::to_string(rowCount) + "x" + std::to_string(hiddenWidth) +
                            " captured=" + std::to_string(capturedCount) + "x" + std::to_string(capturedWidth),
                        joined(rows));
    } catch (const StageLinkError &) {
      break;
    } catch (const std::exception &error) {
      {
        const std::lock_guard lock(mutex_);
        opened_ = nullptr;
      }
      try {
        answerFailure(socket, error.what());
      } catch (const StageLinkError &) {
        break;
      }
    }
  }
  // The stage after went away: the requests it opened go with it.
  const std::lock_guard lock(mutex_);
  for (const uint64_t id : opened) drop(id);
}

void StageServant::open(uint64_t requestId) {
  if (live_.contains(requestId)) return;
  const StateAdmission admission = runtime_.beginStage(requestId);
  if (!admission.granted()) throw std::runtime_error("the previous stage has no free lane");
  cache_.beginRequest(requestId);
  live_.insert(requestId);
  if (opened_) opened_->insert(requestId);
}

void StageServant::drop(uint64_t requestId) noexcept {
  if (!live_.erase(requestId)) return;
  runtime_.end(requestId);
  cache_.endRequest(requestId);
}

ModelBatchItem StageServant::pages(uint64_t requestId, uint64_t tokens) {
  const TokenAdmission admission = cache_.ensureTokens(requestId, tokens);
  if (!admission.granted()) throw std::runtime_error("the previous stage has no K/V pages for the request");
  const PageTableView view = cache_.cutPageTable(requestId);
  ModelBatchItem item;
  item.requestId = requestId;
  item.pageTable = view.pages;
  item.pageTableRevision = view.revision;
  item.pageTableFirstChanged = view.firstChanged;
  item.pageSources = view.sources;
  return item;
}

model::StageRows StageServant::prefill(StageReader &request) {
  const uint64_t id = request.u64();
  const uint64_t position = request.u64();
  const uint32_t count = request.u32();
  const std::vector<uint32_t> history = request.words();
  std::vector<model::StageCaptureSpan> spans(request.u32());
  for (model::StageCaptureSpan &span : spans) {
    span.begin = request.u32();
    span.end = request.u32();
    if (span.begin >= span.end || span.begin < position || span.end > position + count)
      throw std::invalid_argument("a capture span lies outside its prefill");
  }
  if (!count || history.size() != position + count) throw std::invalid_argument("invalid stage prefill");
  // A sequence that does not stop at `position` with the request's history
  // reads it again from the start.
  if (live_.contains(id)) {
    const std::span<const uint32_t> held = runtime_.stageTokens(id);
    if (held.size() != position || !std::ranges::equal(held, std::span(history).first(position))) drop(id);
  }
  if (!live_.contains(id)) {
    open(id);
    if (position) {
      const ModelBatchItem item = pages(id, position);
      static_cast<void>(runtime_.stagePrefill(id, std::span(history).first(position), item, {}));
    }
  }
  const ModelBatchItem item = pages(id, position + count);
  model::StageRows rows = runtime_.stagePrefill(id, std::span(history).subspan(position), item, spans);
  rowsOf_ = count;
  capturedOf_ = 0;
  for (const model::StageCaptureSpan &span : spans) capturedOf_ += span.end - span.begin;
  return rows;
}

model::StageRows StageServant::verify(StageReader &request) {
  const uint32_t width = request.u32();
  if (!width || width > model::ExecutionLimits::maximumBatchWidth) throw std::invalid_argument("invalid stage verify");
  std::vector<model::StageVerifyLane> lanes(width);
  std::vector<std::vector<uint32_t>> tokens(width);
  std::vector<ModelBatchItem> items(width);
  for (uint32_t lane = 0; lane < width; ++lane) {
    model::StageVerifyLane &entry = lanes[lane];
    entry.requestId = request.u64();
    entry.position = request.u64();
    entry.retainedBefore = request.u32();
    entry.verifyRows = request.u32();
    tokens[lane] = request.words();
    entry.tokens = tokens[lane];
    if (!live_.contains(entry.requestId)) throw std::invalid_argument("a stage verify of a request not held");
    runtime_.stageCommit(entry.requestId, entry.retainedBefore);
  }
  for (uint32_t lane = 0; lane < width; ++lane)
    items[lane] = pages(lanes[lane].requestId, lanes[lane].position + tokens[lane].size());
  model::StageRows rows = runtime_.stageVerify(lanes, items);
  rowsOf_ = uint64_t{width} * model::ExecutionLimits::targetVerifyRows;
  capturedOf_ = rowsOf_;
  return rows;
}

void StageServant::end(StageReader &request) {
  const uint64_t id = request.u64();
  static_cast<void>(request.u32());
  drop(id);
  rowsOf_ = capturedOf_ = 0;
}

} // namespace splash::engine
