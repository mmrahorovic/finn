from qonnx.core.modelwrapper import ModelWrapper
import numpy as np

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig
)

from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.general import (
    ConvertSubToAdd,
    ConvertDivToMul,
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
    GiveUniqueParameterTensors,
    RemoveStaticGraphInputs,
    SortGraph,
)

from finn.transformation.streamline.collapse_repeated import (
    CollapseRepeatedAdd,
    CollapseRepeatedMul
)

from finn.transformation.streamline.absorb import (
    AbsorbAddIntoMultiThreshold,
    AbsorbMulIntoMultiThreshold,
    FactorOutMulSignMagnitude,
    #Absorb1BitMulIntoMatMul,
    #Absorb1BitMulIntoConv,
    AbsorbConsecutiveTransposes,
    AbsorbSignBiasIntoMultiThreshold,
    #AbsorbTransposeIntoMultiThreshold,
)

from finn.transformation.streamline.reorder import (
    MoveAddPastMul,
    #MoveScalarMulPastMatMul,
    #MoveScalarAddPastMatMul,
    MoveAddPastConv,
    MoveScalarMulPastConv,
    MoveMulPastFork,
    MoveAddPastFork,
    MoveLinearPastEltwiseAdd,
    MoveTransposeBeforeFork,
    MoveTransposePastMultiThreshold,
    #MoveScalarLinearPastInvariants,
    #MoveMaxPoolPastMultiThreshold,
)

from qonnx.transformation.double_to_single_float import DoubleToSingleFloat
from qonnx.core.datatype import DataType
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.batchnorm_to_affine import BatchNormToAffine

from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_data_layouts import InferDataLayouts

import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul


def step_deeprx_tidy(model: ModelWrapper, cfg: DataflowBuildConfig):
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(GiveUniqueParameterTensors())
    model = model.transform(InferShapes())
    model = model.transform(FoldConstants())
    model = model.transform(RemoveStaticGraphInputs())
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(InferDataTypes())
    return model

def step_deeprx_streamline(model: ModelWrapper, cfg: DataflowBuildConfig):
    streamline_transformations = [
        AbsorbSignBiasIntoMultiThreshold(),
        ConvertSubToAdd(),
        ConvertDivToMul(),
        RemoveIdentityOps(),
        BatchNormToAffine(),
        CollapseRepeatedMul(),
        MoveAddPastMul(),
        #MoveScalarAddPastMatMul(),
        #MoveAddPastConv(),
        #MoveScalarMulPastMatMul(),
        MoveScalarMulPastConv(),
        #MoveScalarLinearPastInvariants(),
        #MoveAddPastMul(),
        CollapseRepeatedAdd(),
        AbsorbAddIntoMultiThreshold(),
        FactorOutMulSignMagnitude(),
        #MoveMaxPoolPastMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        #Absorb1BitMulIntoMatMul(),
        #Absorb1BitMulIntoConv(),
        #RoundAndClipThresholds(),
        MoveAddPastFork(),
        MoveMulPastFork(),
        MoveLinearPastEltwiseAdd(),
        MoveAddPastConv(),
        MoveScalarMulPastConv(),
        AbsorbAddIntoMultiThreshold(),
        AbsorbMulIntoMultiThreshold(),
        MoveAddPastMul(), ## Final layers clean-up
        CollapseRepeatedMul(),
        RemoveUnusedTensors(),
        GiveReadableTensorNames(),
        InferDataTypes(),
        SortGraph(),
        DoubleToSingleFloat(),
        RoundAndClipThresholds()
    ]
    for trn in streamline_transformations:
        model = model.transform(trn)
        model = model.transform(GiveUniqueNodeNames())
    return model

def step_deeprx_convert_to_hls(model: ModelWrapper, cfg: DataflowBuildConfig):
    model.set_tensor_datatype(model.graph.input[0].name, DataType["UINT8"])
    model = model.transform(InferDataLayouts())

    model = model.transform(DoubleToSingleFloat())
    model = model.transform(InferDataTypes())
    model = model.transform(SortGraph())

    to_hls_transformations = [
        LowerConvsToMatMul(),
        AbsorbConsecutiveTransposes(),
        MoveTransposeBeforeFork(),
        MoveTransposePastMultiThreshold(),
        AbsorbConsecutiveTransposes(),
        #to_hls.InferConvInpGen(),
        #RoundAndClipThresholds(),
        #to_hls.InferThresholdingLayer(),
        #to_hls.InferVectorVectorActivation(),
        #to_hls.InferQuantizedMatrixVectorActivation(),
        #to_hls.InferAddStreamsLayer(),
        #to_hls.InferDuplicateStreamsLayer(),
        #to_hls.InferChannelwiseLinearLayer(),
    ]
    for trn in to_hls_transformations:
        model = model.transform(trn)
        model = model.transform(InferDataLayouts())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(InferDataTypes())

    #model = model.transform(RemoveCNVtoFCFlatten())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(RemoveUnusedTensors())
    model = model.transform(SortGraph())

    return model   