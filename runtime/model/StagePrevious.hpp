#pragma once

// The stage before a target that holds a slice of the model's layers which
// does not begin the model (QwenTargetDimensions::embedsTokens false): it runs
// the layers before the slice and hands over the residual rows they leave,
// the rows the slice's first layer reads in place of the token embedding.
// The runtime of the last stage asks it for each prefill part's rows and for
// each decode cycle's verify rows (engine::StageClient carries the requests
// to a servant process, engine::StageServant).

#include <cstdint>
#include <span>
#include <vector>

namespace splash::model {

// Positions [begin, end) of a request whose captured rows (the outputs of
// the layers a DFlash2 draft reads, QwenTargetGeometry::captureLayers) the
// stage after asks for beside its residual rows.
struct StageCaptureSpan final {
  uint32_t begin = 0;
  uint32_t end = 0;
};

// Rows a previous stage computed, as f32: `hidden` the residual rows its last
// layer leaves, row-major of the hidden width; `captured` the captured rows
// of the capture slots its layers hold, row-major of the captured width
// (QwenTargetGeometry::capturedHiddenSize), the other slots' columns zero.
struct StageRows final {
  std::vector<float> hidden;
  std::vector<float> captured;
};

// One lane of a decode cycle: its request, the position of its anchor row,
// the rows of its previous cycle its acceptance retained (none after a
// prefill), the rows of its block the cycle verifies (a MoE target's steps
// route those alone), and the cycle's input tokens, the anchor and the
// draft's proposals.
struct StageVerifyLane final {
  uint64_t requestId = 0;
  uint64_t position = 0;
  uint32_t retainedBefore = 0;
  uint32_t verifyRows = 0;
  std::span<const uint32_t> tokens;
};

class StagePrevious {
public:
  virtual ~StagePrevious() = default;
  // The rows [position, position + count) of a request whose tokens are
  // `history` (at least position + count of them), and the captured rows of
  // `captures`, in order.
  [[nodiscard]] virtual StageRows prefill(uint64_t requestId, uint64_t position, uint32_t count,
                                          std::span<const uint32_t> history,
                                          std::span<const StageCaptureSpan> captures) = 0;
  // Every lane's verify rows, lane after lane, and their captured rows.
  [[nodiscard]] virtual StageRows verify(std::span<const StageVerifyLane> lanes) = 0;
  // The request ended, the rows of its last cycle it retained with it.
  virtual void end(uint64_t requestId, uint32_t retainedBefore) noexcept = 0;
};

} // namespace splash::model
