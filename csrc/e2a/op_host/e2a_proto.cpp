/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace {
constexpr int ATTR_ENUM_BATCH_SIZE = 0;
constexpr int ATTR_ENUM_HIDDEN_SIZE = 1;
constexpr int ATTR_ENUM_EP_RANK_SIZE = 3;
constexpr int ATTR_ENUM_ATTN_RANK_SIZE = 4;
constexpr int ATTR_ENUM_RANK = 5;

constexpr int TWO_DIMS = 2;
}

namespace ge {
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        auto attrPointers = context->GetAttrs();
        int batchSize = *(attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE));
        int hiddenSize = *(attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE));
        int expertRankSize = *(attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE));
        int rank = *(attrPointers->GetInt(ATTR_ENUM_RANK));

        int attentionRankSize = *(attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE));

        gert::Shape* xShape = context->GetOutputShape(0);
        xShape->SetDimNum(TWO_DIMS);
        if (rank < expertRankSize) {
            xShape->SetDim(0, 1);
            xShape->SetDim(1, 1);
        } else if (attentionRankSize <= expertRankSize) {
            xShape->SetDim(0, batchSize);
            xShape->SetDim(1, hiddenSize);
        } else {
            xShape->SetDim(0, batchSize / (attentionRankSize / expertRankSize));
            xShape->SetDim(1, hiddenSize);
        }

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto expandXDType = context->GetInputDataType(0);
        context->SetOutputDataType(0, expandXDType);
        return ge::GRAPH_SUCCESS;
    }
    IMPL_OP(E2a).InferShape(InferShape).InferDataType(InferDataType);
}
