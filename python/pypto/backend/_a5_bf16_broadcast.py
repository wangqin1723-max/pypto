# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device lowering for the A5 two-consumer BF16 C2V FIFO."""

from pypto.pypto_core import ir

MARKER = "// pypto.a5_c2v_bf16_broadcast"

_INITIALIZERS = {
    "system.aic_initialize_pipe": ir.FunctionType.AIC,
    "system.aiv_initialize_pipe": ir.FunctionType.AIV,
}
_TRANSFERS = {"tile.tpush_to_aiv", "tile.tpop_from_aic", "tile.tfree_to_aic"}


class _PipeCalls(ir.IRVisitor):
    def __init__(self, func: ir.Function) -> None:
        super().__init__()
        self.func = func
        self.initializers: list[ir.Call] = []
        self.transfers: list[ir.Call] = []
        self.visit_stmt(func.body)

    def visit_call(self, op: ir.Call) -> None:
        name = op.op.name
        if name in _INITIALIZERS:
            self.initializers.append(op)
        elif name in _TRANSFERS:
            self.transfers.append(op)
        super().visit_call(op)


def _shape(typ: ir.Type) -> tuple[int, ...]:
    if isinstance(typ, ir.BufferType):
        return tuple(typ.shape)
    if isinstance(typ, ir.TileType) and all(isinstance(dim, ir.ConstInt) for dim in typ.shape):
        return tuple(dim.value for dim in typ.shape)
    raise ValueError("C2V broadcast requires static physical tile shapes")


def validate(members: list[ir.Function], pto_code: str) -> bool:
    """Check the one-pipe module contract before enabling the device wrapper."""
    if MARKER not in pto_code:
        return False
    calls = [_PipeCalls(func) for func in members]
    header = pto_code.partition("\n")[0]
    if 'pto.target_arch = "a5"' not in header or 'pto.backend = "vpto"' in header:
        raise ValueError("C2V broadcast requires the A5 PTO backend")
    if len(calls) != 2 or {lane.func.func_type for lane in calls} != set(_INITIALIZERS.values()):
        raise ValueError("C2V broadcast requires a mixed kernel with exactly one AIC and one AIV function")

    endpoints = {}
    for lane in calls:
        if len(lane.initializers) != 1:
            raise ValueError("C2V broadcast requires exactly one pipe per function")
        init = lane.initializers[0]
        if init.kwargs.get("broadcast") is not True or _INITIALIZERS[init.op.name] != lane.func.func_type:
            raise ValueError("C2V broadcast requires matching producer and consumer initializers")
        if init.kwargs.get("ptoas_plan_memory") and not lane.func.attrs.get("ptoas_plan_memory"):
            raise ValueError("ptoas_plan_memory requires both mixed-kernel members to use PTOAS planning")
        if any(op.kwargs.get("split", 0) != 1 for op in lane.transfers):
            raise ValueError("C2V broadcast transfers require split=1 for two-lane FIFO credits")
        shapes = []
        for op in lane.transfers:
            if op.op.name == "tile.tpush_to_aiv":
                shapes.append(_shape(op.args[0].type))
            elif op.op.name == "tile.tpop_from_aic":
                shapes.append(_shape(op.type))
        endpoints[lane.func.func_type] = (
            init.kwargs.get("id", 0),
            init.kwargs["slot_size"],
            init.kwargs.get("slot_num"),
            init.kwargs.get("ptoas_plan_memory", False),
            tuple(shapes),
        )
    producer = endpoints[ir.FunctionType.AIC]
    consumer = endpoints[ir.FunctionType.AIV]
    if not producer[-1] or producer != consumer:
        raise ValueError("C2V broadcast requires matching slot configuration and push/pop shapes")
    return True


# PTOAS inserts ordinary FIFO synchronization using the full-size pop type.
# Its FixPipe epilogue supports one consumer and one static payload shape.
# Use the existing two-lane FIFO credits and send the entire Acc to each UB.
HEADER = r"""
#if defined(__CPU_SIM) || defined(__COSTMODEL)
#error "A5 BF16 C2V broadcast requires A5 device code"
#endif

namespace pto {
template <typename Pipe, typename Acc, TileSplitAxis Split>
__aicore__ inline void PyptoA5BroadcastPush(Pipe& pipe, Acc& acc) {
    static_assert(Pipe::is_c2v_ub && Split == TileSplitAxis::TILE_UP_DOWN);
    static_assert(Acc::Loc == TileType::Acc && std::is_same_v<typename Acc::DType, float>);
#if defined(__DAV_CUBE__)
    using FIFO = typename Pipe::RingFiFo;
    using Payload = Tile<TileType::Vec, bfloat16_t, Acc::Rows, Acc::Cols, BLayout::RowMajor, -1, -1>;
    static_assert(Acc::Rows * Acc::Cols * sizeof(bfloat16_t) <= FIFO::SLOT_SIZE);
    if (pipe.prod.getAllocateStatus() && Pipe::shouldWaitFree(pipe.prod.tileIndex)) {
        pipe.prod.template allocate<Split>();
    }
    const uint64_t address = pipe.fifo.C2V_CONSUMER_BUF +
        (pipe.prod.tileIndex % FIFO::SLOT_NUM) * FIFO::SLOT_SIZE + pipe.prod.entryOffset;
    Payload payload(acc.GetValidRow(), acc.GetValidCol());
    TASSIGN(payload, address);
    TMOV<Payload, Acc, AccToVecMode::SingleModeVec0>(payload, acc);
    TMOV<Payload, Acc, AccToVecMode::SingleModeVec1>(payload, acc);
    ++pipe.prod.tileIndex;
    if (pipe.prod.getRecordStatus()) {
        pipe.prod.template record<Split>();
    }
#else
    (void)pipe;
    (void)acc;
#endif
}

}  // namespace pto

#define TPUSH PyptoA5BroadcastPush
"""

FOOTER = """
#undef TPUSH
"""
