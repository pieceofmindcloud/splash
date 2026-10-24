#pragma once

#include "model/ModelFactory.hpp"
#include "model/StagePrevious.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace splash::model {

class RuntimeInstrumentation;

class Runtime final : public RuntimeModel {
public:
  explicit Runtime(RuntimeContext context);
  ~Runtime() override;
  void checkHealth() override;
  [[nodiscard]] uint32_t statesToActivate() const noexcept override;
  [[nodiscard]] uint32_t prefillGroupRows(bool spills) const noexcept override;
  [[nodiscard]] uint32_t prefillPasses() const noexcept override;

  Runtime(const Runtime &) = delete;
  Runtime &operator=(const Runtime &) = delete;

  // Direct native-oracle entry point. Production admission uses
  // begin() and installs its cache-aware plan explicitly.
  void beginColdRequest(const ModelRequest &request, uint32_t stateLane);
  [[nodiscard]] StateAdmission
  begin(const ModelRequest &request) override;
  void suspend(uint64_t requestId) override;
  [[nodiscard]] StateAdmission
  resume(const ModelRequest &request) override;
  [[nodiscard]] std::unique_ptr<StateRestore> beginRestore(
      uint64_t requestId, uint32_t boundary,
      std::shared_ptr<const CompositeState> state, bool restoreDraft,
      std::function<void()> completion) override;
  void setDraftContextPlan(uint64_t requestId, DraftContextPlan plan) override;
  [[nodiscard]] std::vector<ModelStepResult>
  prefill(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  submit(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion) override;
  [[nodiscard]] std::vector<ModelStepResult>
  decode(const BatchPlan &plan, std::span<const ModelBatchItem> items);
  [[nodiscard]] std::shared_ptr<StateReservation>
  reserveState(uint64_t requestId, uint32_t tokens, metal::AllocationFailure &refusal) override;
  [[nodiscard]] std::shared_ptr<const CompositeState>
  snapshot(uint64_t requestId) override;
  [[nodiscard]] uint64_t snapshotBytes() const noexcept override;
  [[nodiscard]] bool canSnapshotToDisk(bool inUse) const noexcept override;
  [[nodiscard]] std::unique_ptr<StateOffload>
  snapshotToDisk(uint64_t requestId, std::function<void()> completion) override;
  [[nodiscard]] ModelMemoryAdmission reserveSpilledReads(uint64_t requestId) override;
  [[nodiscard]] ModelMemoryAdmission reserveHistoryStaging(uint64_t requestId, uint32_t pages) override;
  [[nodiscard]] ModelMemoryAdmission growSpilledReads() override;
  [[nodiscard]] uint64_t reclaimIdleState(bool keepLane,
                                          IdleMemory scope) noexcept override;
  [[nodiscard]] std::optional<std::string>
  provideMask(uint64_t requestId, std::span<const uint32_t> words) override;
  void end(uint64_t requestId) override;

  [[nodiscard]] WarmupStepResult warmupPrefill(uint32_t rows) override;
  [[nodiscard]] WarmupStepResult
  warmupDecodeBatch(uint32_t width) override;
  [[nodiscard]] WarmupStepResult
  warmupCompositeStateRestore() override;
  [[nodiscard]] ModelMemoryActual
  actualRuntimeMemory() const override;

  [[nodiscard]] ModelTelemetry
  telemetry() const noexcept override;

  // A stage servant's sequences (engine::StageServant): the runtime of a
  // target whose layers do not end the model computes, for the stage after
  // it, the residual rows its last layer leaves. A sequence has a lane and
  // its K/V pages, which its caller keeps (`pages`, for its rows); it never
  // drafts nor selects.
  //
  // Begins sequence `requestId` at a free lane, or refuses as begin() does.
  [[nodiscard]] StateAdmission beginStage(uint64_t requestId);
  // The tokens the sequence has read, its committed length.
  [[nodiscard]] std::span<const uint32_t> stageTokens(uint64_t requestId);
  // Reads `tokens` at the sequence's end, in commands of one prefill chunk,
  // and returns their rows and the captured rows of `captures` (inside
  // them, in order); `pages` the sequence's pages for them all.
  [[nodiscard]] StageRows stagePrefill(uint64_t requestId, std::span<const uint32_t> tokens,
                                       const ModelBatchItem &pages, std::span<const StageCaptureSpan> captures);
  // One verify of each lane's tokens at its end (StageVerifyLane: the lane's
  // previous verify was committed with stageCommit), in one command: their
  // rows and captured rows, lane after lane. The rows stay pending until
  // stageCommit() keeps the first `retained` of them; `pages` holds each
  // lane's pages for its rows.
  [[nodiscard]] StageRows stageVerify(std::span<const StageVerifyLane> lanes,
                                      std::span<const ModelBatchItem> pages);
  // Keeps the first `retained` rows of the sequence's last verify.
  void stageCommit(uint64_t requestId, uint32_t retained);

private:
  // The state lane of a resident request; the storage checks that its
  // committed state can be cached.
  [[nodiscard]] uint32_t residentLane(uint64_t requestId);
  void finishRestore(uint64_t requestId, uint32_t boundary, bool restoreDraft);
  void prepareWarmupDecode(uint64_t requestId, uint32_t anchor);
  [[nodiscard]] StateAdmission beginAt(const ModelRequest &request,
                                       uint32_t stateLane);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  prefillAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
               std::function<void()> completion);
  [[nodiscard]] std::unique_ptr<ModelBatchTicket>
  decodeAsync(const BatchPlan &plan, std::span<const ModelBatchItem> items,
              std::function<void()> completion);

  friend class RuntimeInstrumentation;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace splash::model
