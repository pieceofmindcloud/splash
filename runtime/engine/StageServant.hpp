#pragma once

// Both ends of the requests a stage link carries (engine/StageLink.hpp):
// StageClient, the previous stage as the runtime of the stage after it asks
// it for rows (model::StagePrevious), and StageServant, which answers them
// with the runtime of a target whose layers begin the model but do not end
// it (model::Runtime's stage sequences), each request's K/V pages from the
// cache.
//
// Requests, one raw frame each, and their one answer frame:
//   0 hello:   text secret, u32 the asking stage's first layer
//              -> raw: u8 0, u32 first layer, u32 end layer of the servant;
//   1 prefill: u64 request, u64 position, u32 count, words history, u32 spans,
//              then each span's u32 begin and u32 end -> rows;
//   2 verify:  u32 lanes, then each lane's u64 request, u64 position,
//              u32 rows retained before, u32 verify rows, words tokens -> rows;
//   3 end:     u64 request, u32 rows retained before -> raw: u8 0.
// Rows answer as one tensor (StageRows: the hidden rows, then the captured
// rows), its metadata "hidden=ROWSxWIDTH captured=ROWSxWIDTH". A request that
// fails answers a raw frame of u8 1 and its reason. A connection whose first
// request is not a hello with the servant's secret, for a stage that starts
// where the servant ends, is closed.
//
// The servant keeps a request's sequence from its first prefill to its end.
// A request it does not hold, or whose sequence does not stop where a
// prefill starts, reads its history from the start first (a sequence never
// goes back); a request the stage after resumes reads it so.

#include "engine/StageLink.hpp"
#include "model/Model.hpp"
#include "model/StagePrevious.hpp"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_set>

namespace splash::model {
class Runtime;
} // namespace splash::model

namespace splash::engine {

class Cache;

inline constexpr uint8_t kStageHello = 0;
inline constexpr uint8_t kStagePrefill = 1;
inline constexpr uint8_t kStageVerify = 2;
inline constexpr uint8_t kStageEnd = 3;

class StageClient final : public model::StagePrevious {
public:
  // Says hello on `socket` with `secret`, for a stage whose layers start at
  // `firstLayer`: the servant's must end there.
  StageClient(StageSocket socket, std::string_view secret, uint32_t firstLayer);

  [[nodiscard]] model::StageRows prefill(uint64_t requestId, uint64_t position, uint32_t count,
                                         std::span<const uint32_t> history,
                                         std::span<const model::StageCaptureSpan> captures) override;
  [[nodiscard]] model::StageRows verify(std::span<const model::StageVerifyLane> lanes) override;
  void end(uint64_t requestId, uint32_t retainedBefore) noexcept override;

private:
  // Sends a request and reads its answer: a raw frame, or rows.
  std::vector<uint8_t> call(const StageWriter &request);
  model::StageRows rows(const StageWriter &request);

  std::mutex mutex_;
  StageSocket socket_;
};

class StageServant final {
public:
  // A servant of the model's layers [firstLayer, endLayer), whose
  // connections present `secret`.
  StageServant(model::Runtime &runtime, Cache &cache, std::string secret, uint32_t firstLayer,
               uint32_t endLayer)
      : runtime_(runtime), cache_(cache), secret_(std::move(secret)), firstLayer_(firstLayer),
        endLayer_(endLayer) {}

  // Answers one connection's requests until it closes; connections may be
  // served on threads of their own, one request at a time across them.
  void serve(StageSocket &socket);

private:
  model::StageRows prefill(StageReader &request);
  model::StageRows verify(StageReader &request);
  void end(StageReader &request);
  // The request's sequence, begun when the servant does not hold it.
  void open(uint64_t requestId);
  void drop(uint64_t requestId) noexcept;
  // The request's page list for its first `tokens` tokens.
  [[nodiscard]] ModelBatchItem pages(uint64_t requestId, uint64_t tokens);

  model::Runtime &runtime_;
  Cache &cache_;
  const std::string secret_;
  const uint32_t firstLayer_;
  const uint32_t endLayer_;
  std::mutex mutex_;
  std::unordered_set<uint64_t> live_;
  // The requests the connection being served opened, which go with it.
  std::unordered_set<uint64_t> *opened_ = nullptr;
  // The rows and captured rows of the last answer.
  uint64_t rowsOf_ = 0;
  uint64_t capturedOf_ = 0;
};

} // namespace splash::engine
