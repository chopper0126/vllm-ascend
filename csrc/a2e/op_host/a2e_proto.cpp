/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "log/ops_log.h"
#include "error/ops_error.h"
#include "graph/utils/type_utils.h"
#include "register/op_def_registry.h"

namespace ge {
    constexpr int ATTR_ENUM_BATCH_SIZE = 0;
    constexpr int ATTR_ENUM_HIDDEN_SIZE = 1;
    constexpr int ATTR_ENUM_TOPK = 2;
    constexpr int ATTR_ENUM_EP_RANK_SIZE = 3;
    constexpr int ATTR_ENUM_ATTN_RANK_SIZE = 4;
    constexpr int ATTR_ENUM_RANK = 5;

    constexpr int INPUT_EXPANDX_IDX = 0;
    constexpr int INPUT_EXPERT_SCALES_IDX = 2;
    constexpr int INPUT_EXPERT_IDS_IDX = 1;
    static ge::graphStatus InferShape(gert::InferShapeContext* context)
    {
        auto attrPointers = context->GetAttrs();
        int batchSize = *(attrPointers->GetInt(ATTR_ENUM_BATCH_SIZE));
        int hiddenSize = *(attrPointers->GetInt(ATTR_ENUM_HIDDEN_SIZE));
        int topk = *(attrPointers->GetInt(ATTR_ENUM_TOPK));
        int expertRankSize = *(attrPointers->GetInt(ATTR_ENUM_EP_RANK_SIZE));
        int attentionRankSize = *(attrPointers->GetInt(ATTR_ENUM_ATTN_RANK_SIZE));
        int rank = *(attrPointers->GetInt(ATTR_ENUM_RANK));
        batchSize = batchSize * (attentionRankSize + expertRankSize - 1) / expertRankSize;
    
        gert::Shape* expandXShape = context->GetOutputShape(0);
        expandXShape->SetDimNum(2);
        if (rank < expertRankSize) {
            expandXShape->SetDim(0, batchSize);
            expandXShape->SetDim(1, hiddenSize);
        } else {
            expandXShape->SetDim(0, 1);
            expandXShape->SetDim(1, 1);
        }

        gert::Shape* simulateExpertIdsShape = context->GetOutputShape(1);
        simulateExpertIdsShape->SetDimNum(2);
        if (rank < attentionRankSize) {
            simulateExpertIdsShape->SetDim(0, batchSize);
            simulateExpertIdsShape->SetDim(1, topk);
        } else {
            simulateExpertIdsShape->SetDim(0, 1);
            simulateExpertIdsShape->SetDim(1, 1);
        }

        gert::Shape* simulateExpertScalesShape = context->GetOutputShape(2);
        simulateExpertScalesShape->SetDimNum(2);
        if (rank < attentionRankSize) {
            simulateExpertScalesShape->SetDim(0, batchSize);
            simulateExpertScalesShape->SetDim(1, topk);
        } else {
            simulateExpertScalesShape->SetDim(0, 1);
            simulateExpertScalesShape->SetDim(1, 1);
        }

        gert::Shape* attenBatchSizeShape = context->GetOutputShape(3);
        attenBatchSizeShape->SetDimNum(1);
        attenBatchSizeShape->SetDim(0, (attentionRankSize + expertRankSize - 1) / expertRankSize);

        gert::Shape* xActiveMaskOutShape = context->GetOutputShape(4);
        xActiveMaskOutShape->SetDimNum(1);
        if (rank < attentionRankSize) {
            xActiveMaskOutShape->SetDim(0, batchSize);
        } else {
            xActiveMaskOutShape->SetDim(0, 1);
        }

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
    {
        const auto expertIdsType = context->GetInputDataType(INPUT_EXPERT_IDS_IDX);
        const auto expertScalesType = context->GetInputDataType(INPUT_EXPERT_SCALES_IDX);
        const auto expandXDType = context->GetInputDataType(INPUT_EXPANDX_IDX);

        int outputIdx = 0;
        context->SetOutputDataType(outputIdx++, expandXDType);
        context->SetOutputDataType(outputIdx++, expertIdsType);
        context->SetOutputDataType(outputIdx++, expertScalesType);
        context->SetOutputDataType(outputIdx++, ge::DT_INT32);
        context->SetOutputDataType(outputIdx++, ge::DT_BOOL);
        return ge::GRAPH_SUCCESS;
    }
    IMPL_OP(A2e).InferShape(InferShape).InferDataType(InferDataType);
}
