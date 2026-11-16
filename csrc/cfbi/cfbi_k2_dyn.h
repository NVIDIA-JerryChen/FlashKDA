// Dynamic-schedule K2 (cfbi_k2_dyn.cu): host entry points shared by the launcher (cfbi_launch.cu) and the translation unit that holds the kernels.
#pragma once

#include "cfbi/cfbi_common.cuh"

namespace cfbi {

// Builds the schedule of a call into the plan buffer, behind the (N + 1) SeqPlan entries `plan` (a single-CTA kernel on `stream`; needs the plan of the
// call). sms: SMs of the device the K2 stage will run on; k2_bv: 0 = the device policy, 8 / 16 / 32 = forced width of every pair;
// call_fixed: the call has no cu_seqlens (the old schedule is the closed form of the fixed-length launcher over all N * H pair positions: the builder compares
// against it when the active chunk counts are equal and writes it as it is when they are not).
void k2_sched_build(SeqPlan* plan, int N, int H, int sms, int k2_bv, bool call_fixed, cudaStream_t stream);

// The K2 stage of the dynamic-schedule implementation: when the old schedule is the only one that can matter (everything fits one wave of the SMs at
// 8 columns, nothing to order or to choose) the K2 kernel decodes the old schedule in closed form, otherwise the schedule builder runs first and the
// K2 kernel reads its table. Same inputs as the other K2 launches (p.plan holds the plan of the call).
void launch_k2_dyn(const Params& p, const ForwardArgs& a, int sms, cudaStream_t stream);

}  // namespace cfbi
