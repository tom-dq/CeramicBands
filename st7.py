# Friendly wrapper around the Strand7 API.

import St7API
import tempfile
import dataclasses
import enum
import typing
import ctypes
import pathlib
import collections

import numpy

T_Path = typing.Union[pathlib.Path, str]

_ARRAY_NAME = "array_name"

def chk(iErr):
    if iErr != 0:
        raise Exception(iErr)


class Entity(enum.Enum):
    tyNODE = St7API.tyNODE
    tyBEAM = St7API.tyBEAM
    tyPLATE = St7API.tyPLATE
    tyBRICK = St7API.tyBRICK
    tyLINK = St7API.tyLINK
    tyVERTEX = St7API.tyVERTEX
    tyGEOMETRYEDGE = St7API.tyGEOMETRYEDGE
    tyGEOMETRYFACE = St7API.tyGEOMETRYFACE
    tyLOADPATH = St7API.tyLOADPATH
    tyGEOMETRYCOEDGE = St7API.tyGEOMETRYCOEDGE
    tyGEOMETRYLOOP = St7API.tyGEOMETRYLOOP


class SolverType(enum.Enum):
    stLinearStatic = St7API.stLinearStatic
    stLinearBuckling = St7API.stLinearBuckling
    stNonlinearStatic = St7API.stNonlinearStatic
    stNaturalFrequency = St7API.stNaturalFrequency
    stHarmonicResponse = St7API.stHarmonicResponse
    stSpectralResponse = St7API.stSpectralResponse
    stLinearTransientDynamic = St7API.stLinearTransientDynamic
    stNonlinearTransientDynamic = St7API.stNonlinearTransientDynamic
    stSteadyHeat = St7API.stSteadyHeat
    stTransientHeat = St7API.stTransientHeat
    stLoadInfluence = St7API.stLoadInfluence
    stQuasiStatic = St7API.stQuasiStatic


class SolverMode(enum.Enum):
    smNormalRun = St7API.smNormalRun
    smNormalCloseRun = St7API.smNormalCloseRun
    smProgressRun = St7API.smProgressRun
    smBackgroundRun = St7API.smBackgroundRun


class PreLoadType(enum.Enum):
    plPlatePreStrain = St7API.plPlatePreStrain
    plPlatePreStress = St7API.plPlatePreStress


class NodeResultType(enum.Enum):
    rtNodeDisp = St7API.rtNodeDisp
    rtNodeVel = St7API.rtNodeVel
    rtNodeAcc = St7API.rtNodeAcc
    rtNodePhase = St7API.rtNodePhase
    rtNodeReact = St7API.rtNodeReact
    rtNodeTemp = St7API.rtNodeTemp
    rtNodeFlux = St7API.rtNodeFlux
    rtNodeInfluence = St7API.rtNodeInfluence
    rtNodeInertia = St7API.rtNodeInertia


class PlateResultType(enum.Enum):
    rtPlateStress = St7API.rtPlateStress
    rtPlateStrain = St7API.rtPlateStrain
    rtPlateEnergy = St7API.rtPlateEnergy
    rtPlateForce = St7API.rtPlateForce
    rtPlateMoment = St7API.rtPlateMoment
    rtPlateCurvature = St7API.rtPlateCurvature
    rtPlatePlyStress = St7API.rtPlatePlyStress
    rtPlatePlyStrain = St7API.rtPlatePlyStrain
    rtPlatePlyReserve = St7API.rtPlatePlyReserve
    rtPlateFlux = St7API.rtPlateFlux
    rtPlateGradient = St7API.rtPlateGradient
    rtPlateRCDesign = St7API.rtPlateRCDesign
    rtPlateCreepStrain = St7API.rtPlateCreepStrain
    rtPlateSoil = St7API.rtPlateSoil
    rtPlateUser = St7API.rtPlateUser
    rtPlateNodeReact = St7API.rtPlateNodeReact
    rtPlateNodeDisp = St7API.rtPlateNodeDisp
    rtPlateNodeBirthDisp = St7API.rtPlateNodeBirthDisp
    rtPlateEffectiveStress = St7API.rtPlateEffectiveStress
    rtPlateEffectiveForce = St7API.rtPlateEffectiveForce
    rtPlateNodeFlux = St7API.rtPlateNodeFlux
    rtPlateTotalStrain = St7API.rtPlateTotalStrain
    rtPlateTotalCurvature = St7API.rtPlateTotalCurvature


class PlateResultSubType(enum.Enum):
    stPlateLocal = St7API.stPlateLocal
    stPlateGlobal = St7API.stPlateGlobal
    stPlateCombined = St7API.stPlateCombined
    stPlateSupport = St7API.stPlateSupport
    stPlateDevLocal = St7API.stPlateDevLocal
    stPlateDevGlobal = St7API.stPlateDevGlobal
    stPlateDevCombined = St7API.stPlateDevCombined


class SampleLocation(enum.Enum):
    spCentroid = St7API.spCentroid
    spGaussPoints = St7API.spGaussPoints
    spNodesAverageNever = St7API.spNodesAverageNever
    spNodesAverageAll = St7API.spNodesAverageAll
    spNodesAverageSame = St7API.spNodesAverageSame


class PlateSurface(enum.Enum):
    psPlateMidPlane = St7API.psPlateMidPlane
    psPlateZMinus = St7API.psPlateZMinus
    psPlateZPlus = St7API.psPlateZPlus


class SolverDefaultLogical(enum.Enum):
    spDoSturm = St7API.spDoSturm
    spNonlinearMaterial = St7API.spNonlinearMaterial
    spNonlinearGeometry = St7API.spNonlinearGeometry
    spAddKg = St7API.spAddKg
    spCalcDampingRatios = St7API.spCalcDampingRatios
    spIncludeLinkReactions = St7API.spIncludeLinkReactions
    spFullSystemTransient = St7API.spFullSystemTransient
    spNonlinearHeat = St7API.spNonlinearHeat
    spLumpedLoadBeam = St7API.spLumpedLoadBeam
    spLumpedLoadPlate = St7API.spLumpedLoadPlate
    spLumpedMassBeam = St7API.spLumpedMassBeam
    spLumpedMassPlate = St7API.spLumpedMassPlate
    spLumpedMassBrick = St7API.spLumpedMassBrick
    spForceDrillCheck = St7API.spForceDrillCheck
    spSaveRestartFile = St7API.spSaveRestartFile
    spSaveIntermediate = St7API.spSaveIntermediate
    spExcludeMassX = St7API.spExcludeMassX
    spExcludeMassY = St7API.spExcludeMassY
    spExcludeMassZ = St7API.spExcludeMassZ
    spSaveSRSSSpectral = St7API.spSaveSRSSSpectral
    spSaveCQCSpectral = St7API.spSaveCQCSpectral
    spDoResidualsCheck = St7API.spDoResidualsCheck
    spSuppressAllSingularities = St7API.spSuppressAllSingularities
    spReducedLogFile = St7API.spReducedLogFile
    spIncludeRotationalMass = St7API.spIncludeRotationalMass
    spIgnoreCompressiveBeamKg = St7API.spIgnoreCompressiveBeamKg
    spAutoScaleKg = St7API.spAutoScaleKg
    spScaleSupports = St7API.spScaleSupports
    spAutoShift = St7API.spAutoShift
    spSaveTableInsertedSteps = St7API.spSaveTableInsertedSteps
    spSaveLastRestartStep = St7API.spSaveLastRestartStep
    spDoInstantNTA = St7API.spDoInstantNTA
    spAllowExtraIterations = St7API.spAllowExtraIterations
    spPredictImpact = St7API.spPredictImpact
    spAutoWorkingSet = St7API.spAutoWorkingSet
    spDampingForce = St7API.spDampingForce
    spLimitDisplacementNLA = St7API.spLimitDisplacementNLA
    spLimitRotationNLA = St7API.spLimitRotationNLA
    spSaveFinalSubStep = St7API.spSaveFinalSubStep
    spCablesAsMultiCase = St7API.spCablesAsMultiCase
    spShowMessages = St7API.spShowMessages
    spShowProgress = St7API.spShowProgress
    spShowConvergenceGraph = St7API.spShowConvergenceGraph
    spSpectralBaseExcitation = St7API.spSpectralBaseExcitation
    spSpectralLoadExcitation = St7API.spSpectralLoadExcitation
    spCheckEigenvector = St7API.spCheckEigenvector
    spAppendRemainingTime = St7API.spAppendRemainingTime
    spIncludeFollowerLoadKG = St7API.spIncludeFollowerLoadKG
    spInertiaForce = St7API.spInertiaForce
    spSolverGeneratesCombinations = St7API.spSolverGeneratesCombinations


class SolverDefaultInteger(enum.Enum):
    spTreeStartNumber = St7API.spTreeStartNumber
    spNumFrequency = St7API.spNumFrequency
    spNumBucklingModes = St7API.spNumBucklingModes
    spMaxIterationEig = St7API.spMaxIterationEig
    spMaxIterationNonlin = St7API.spMaxIterationNonlin
    spNumBeamSlicesSpectral = St7API.spNumBeamSlicesSpectral
    spMaxConjugateGradientIter = St7API.spMaxConjugateGradientIter
    spMaxNumWarnings = St7API.spMaxNumWarnings
    spFiniteStrainDefinition = St7API.spFiniteStrainDefinition
    spBeamLength = St7API.spBeamLength
    spFormStiffMatrix = St7API.spFormStiffMatrix
    spMaxUpdateInterval = St7API.spMaxUpdateInterval
    spFormNonlinHeatStiffMatrix = St7API.spFormNonlinHeatStiffMatrix
    spExpandWorkingSet = St7API.spExpandWorkingSet
    spMinNumViscoUnits = St7API.spMinNumViscoUnits
    spMaxNumViscoUnits = St7API.spMaxNumViscoUnits
    spCurveFitTimeUnit = St7API.spCurveFitTimeUnit
    spStaticAutoStepping = St7API.spStaticAutoStepping
    spBeamKgType = St7API.spBeamKgType
    spDynamicAutoStepping = St7API.spDynamicAutoStepping
    spMaxIterationHeat = St7API.spMaxIterationHeat


class TableType(enum.Enum):
    ttVsTime = St7API.ttVsTime
    ttVsTemperature = St7API.ttVsTemperature
    ttVsFrequency = St7API.ttVsFrequency
    ttStressStrain = St7API.ttStressStrain
    ttForceDisplacement = St7API.ttForceDisplacement
    ttMomentCurvature = St7API.ttMomentCurvature
    ttMomentRotation = St7API.ttMomentRotation
    ttAccVsTime = St7API.ttAccVsTime
    ttForceVelocity = St7API.ttForceVelocity
    ttVsPosition = St7API.ttVsPosition
    ttStrainTime = St7API.ttStrainTime


_T_ctypes_type = typing.Union[typing.Type[ctypes.c_long], typing.Type[ctypes.c_double]]

class _InternalSubArrayDefinition(typing.NamedTuple):
    """Represents, for example, the "Integers" argument of St7SetEntityContourSettingsLimits - how many values there are in there, what type they are, etc. """
    elem_type: _T_ctypes_type  # e.g., ctypes.c_long, ctypes.c_double  (not an instance like ctypes.c_long(34)... )
    fields: typing.List[dataclasses.Field]
    array_name_override: str

    @property 
    def array_name(self) -> str:
        if self.array_name_override:
            return self.array_name_override

        lookups = {
            ctypes.c_long: "Integers",
            ctypes.c_double: "Doubles",
        }

        return lookups[self.elem_type]

    @property
    def array_length(self) -> int:
        # Have a buffer on there in case...
        return 10 + max(getattr(St7API, field.name) for field in self.fields)

    def make_empty_array(self):
        array = (self.elem_type * self.array_length)()
        return array

    def instance_from_st7_array(self, ints_or_floats) -> "_InternalSubArrayInstance":
        values = {}

        for field in self.fields:
            idx = getattr(St7API, field.name)
            values[field.name] = field.type(ints_or_floats[idx])  # Convert to int if it's a bool

        return _InternalSubArrayInstance(array_def=self, values=values)


class _InternalSubArrayInstance(typing.NamedTuple):
    """Represents an instance of, say, the "Integers" argument of St7SetEntityContourSettingsLimits populated with values"""
    array_def: _InternalSubArrayDefinition
    values: typing.Dict[str, typing.Union[bool, int, float]]

    def to_st7_array(self):
        working_array = self.array_def.make_empty_array()

        for key, val in self.values.items():
            idx = getattr(St7API, key)
            working_array[idx] = val

        return working_array


@dataclasses.dataclass
class _St7ArrayBase:
    """All those arrays of integers can inherit from this to get convenience conversion functions."""

    # TODO - future plan for functions in which there are multiple arrays of the same type,
    #   like "St7GetBeamPropertyData" with two Doubles arrays:
    #   Support custom field creation like this
    #   ipAREA : float = field(metadata={"array_name": "SectionData"})
    #   ipModulus : float = field(metadata={"array_name": "MaterialData"})

    # Also TODO: support stuff like connection arrays? Or things where there is no ipAAA constant?

    # Another TODO - it would be good to be able to convert to and from enums where they appear in an integer array.

    @classmethod
    def get_sub_arrays(cls) -> typing.Iterable[_InternalSubArrayDefinition]:
        
        def sub_array_key(field: dataclasses.Field):

            if field.type in {int, bool}:
                c_type = ctypes.c_long

            elif field.type == float:
                c_type = ctypes.c_double

            else:
                raise ValueError(field)

            array_name_override = field.metadata.get(_ARRAY_NAME, '')
            
            return c_type, array_name_override

        # Collect the sub-array keys
        sub_array_list = collections.defaultdict(list)

        for field in dataclasses.fields(cls):
            sub_array_list[sub_array_key(field)].append(field)

        for (c_type, array_name_override), fields in sub_array_list.items():
            yield _InternalSubArrayDefinition(elem_type=c_type, fields=fields, array_name_override=array_name_override)


    def get_sub_array_instances(self) -> typing.Iterable[_InternalSubArrayInstance]:
        instance_values = dataclasses.asdict(self)

        for sub_array_def in self.get_sub_arrays():
            this_subarray_instance_values = {}
            for field in sub_array_def.fields:
                key = field.name
                this_subarray_instance_values[key] = instance_values.pop(key)

            yield _InternalSubArrayInstance(array_def=sub_array_def, values=this_subarray_instance_values)

        if instance_values:
            raise ValueError(f"did not find a sub-array for the following: {instance_values}")

    @classmethod
    def get_single_sub_array_of_type(cls, target_type: _T_ctypes_type) -> _InternalSubArrayDefinition:

        all_matches = [sub_array for sub_array in cls.get_sub_arrays() if sub_array.elem_type == target_type]

        if len(all_matches) == 1:
            return all_matches.pop()

        raise ValueError(f"Expected one array of type {target_type} - got {len(all_matches)}: {all_matches}")


    def get_single_sub_array_instance_of_type(self, target_type: _T_ctypes_type) -> _InternalSubArrayInstance:
        all_matches = [sub_array_inst for sub_array_inst in self.get_sub_array_instances() if sub_array_inst.array_def.elem_type == target_type]

        if len(all_matches) == 1:
            return all_matches.pop()

        raise ValueError(f"Expected one array instance of type {target_type} - got {len(all_matches)}: {all_matches}")


    @classmethod
    def instance_from_sub_array_instances(cls, *sub_array_instances: _InternalSubArrayInstance) -> "_St7ArrayBase":
        working_dict = {}
        for sub_array_instance in sub_array_instances:
            working_dict.update(sub_array_instance.values)

        return cls(**working_dict)

    @classmethod
    def from_st7_array(cls, ints_or_floats):
        working_dict = {}

        for field in dataclasses.fields(cls):
            idx = getattr(St7API, field.name)
            working_dict[field.name] = field.type(ints_or_floats[idx])  # Convert to int 

        return cls(**working_dict)


    def to_st7_array(self) -> ctypes.Array:
        name_to_idx = {field.name: getattr(St7API, field.name) for field in dataclasses.fields(self)}

        array = self.make_empty_array()

        for field, value in dataclasses.asdict(self).items():
            idx = name_to_idx[field]
            array[idx] = value

        return array

    @classmethod
    def get_array_length(cls) -> int:
        # Have a buffer on there in case...
        return 10 + max(getattr(St7API, field.name) for field in dataclasses.fields(cls))

    @classmethod
    def get_array_element_type(cls):
        """Returns ctypes element type"""
        all_types = {field.type for field in dataclasses.fields(cls)}

        if all_types <= {int, bool}:
            return ctypes.c_long

        elif all_types == {float}:
            return ctypes.c_double

        else:
            raise ValueError(all_types)






@dataclasses.dataclass
class ContourSettingsStyle(_St7ArrayBase):
    ipContourStyle: int
    ipReverse: bool
    ipSeparator: bool
    ipBand1Colour: int
    ipBand2Colour: int
    ipSeparatorColour: int
    ipLineBackColour: int
    ipMonoColour: int
    ipMinColour: int
    ipMaxColour: int
    ipLimitMin: bool
    ipLimitMax: bool


@dataclasses.dataclass
class ContourSettingsLimit(_St7ArrayBase):
    ipContourLimit: int
    ipContourMode: int
    ipNumContours: int
    ipSetMinLimit: bool
    ipSetMaxLimit: bool

    ipMinLimit: float
    ipMaxLimit: float




class Vector3(typing.NamedTuple):
    x: float
    y: float
    z: float

    def __radd__(self, other):
        # This is just a convenience so I can use "sum" rather than "reduce with operator.add"...
        if isinstance(other, Vector3):
            return self.__add__(other)

        elif isinstance(other, int):
            if other == 0:
                return self

            else:
                raise ValueError("Can only add to a zero int, and I probably shouldn't even be doing that.")

        else:
            raise TypeError(other)


    def __add__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError(other)

        return Vector3(
            x=self.x + other.x,
            y=self.y + other.y,
            z=self.z + other.z,
        )

    def __sub__(self, other):
        if not isinstance(other, Vector3):
            raise TypeError(other)

        return Vector3(
            x=self.x - other.x,
            y=self.y - other.y,
            z=self.z - other.z,
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(other)

        return Vector3(
            x=self.x / other,
            y=self.y / other,
            z=self.z / other,
        )

    def __abs__(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5


class StrainTensor(typing.NamedTuple):
    xx: float
    yy: float
    zz: float
    xy: float
    yz: float
    zx: float

    def __radd__(self, other):
        # This is just a convenience so I can use "sum" rather than "reduce with operator.add"...
        if isinstance(other, StrainTensor):
            return self.__add__(other)

        elif isinstance(other, int):
            if other == 0:
                return self

            else:
                raise ValueError("Can only add to a zero int, and I probably shouldn't even be doing that.")

        else:
            raise TypeError(other)


    def __add__(self, other):
        if not isinstance(other, StrainTensor):
            raise TypeError(other)

        return StrainTensor(
            xx=self.xx + other.xx,
            yy=self.yy + other.yy,
            zz=self.zz + other.zz,
            xy=self.xy + other.xy,
            yz=self.yz + other.yz,
            zx=self.zx + other.zx,
        )

    def __sub__(self, other):
        if not isinstance(other, StrainTensor):
            raise TypeError(other)

        return StrainTensor(
            xx=self.xx - other.xx,
            yy=self.yy - other.yy,
            zz=self.zz - other.zz,
            xy=self.xy - other.xy,
            yz=self.yz - other.yz,
            zx=self.zx - other.zx,
        )

    def __truediv__(self, other):
        if not isinstance(other, (int, float)):
            raise TypeError(other)

        return StrainTensor(
            xx=self.xx / other,
            yy=self.yy / other,
            zz=self.zz / other,
            xy=self.xy / other,
            yz=self.yz / other,
            zx=self.zx / other,
        )

    def __abs__(self):
        # Does the max principal make sense as an "abs" value? Not really sure...
        w = numpy.linalg.eigvals(self.as_np_array())
        return max(w)

    def as_np_array(self) -> numpy.array:
        return numpy.array([
            [self.xx, 0.5*self.xy, 0.5*self.zx],
            [0.5*self.xy, self.yy, 0.5*self.yz],
            [0.5*self.zx, 0.5*self.yz, self.zz],
        ])


class CanvasSize(typing.NamedTuple):
    width: int
    height: int


class ResultOutput(typing.NamedTuple):
    num_points: int
    num_cols: int
    results: typing.Tuple[float]


class BeamContour(enum.Enum):
    ctBeamNone = St7API.ctBeamNone
    ctBeamLength = St7API.ctBeamLength
    ctBeamAxis1 = St7API.ctBeamAxis1
    ctBeamAxis2 = St7API.ctBeamAxis2
    ctBeamAxis3 = St7API.ctBeamAxis3
    ctBeamEA = St7API.ctBeamEA
    ctBeamEI11 = St7API.ctBeamEI11
    ctBeamEI22 = St7API.ctBeamEI22
    ctBeamGJ = St7API.ctBeamGJ
    ctBeamEAFactor = St7API.ctBeamEAFactor
    ctBeamEI11Factor = St7API.ctBeamEI11Factor
    ctBeamEI22Factor = St7API.ctBeamEI22Factor
    ctBeamGJFactor = St7API.ctBeamGJFactor
    ctBeamOffset1 = St7API.ctBeamOffset1
    ctBeamOffset2 = St7API.ctBeamOffset2
    ctBeamStiffnessFactor1 = St7API.ctBeamStiffnessFactor1
    ctBeamStiffnessFactor2 = St7API.ctBeamStiffnessFactor2
    ctBeamStiffnessFactor3 = St7API.ctBeamStiffnessFactor3
    ctBeamStiffnessFactor4 = St7API.ctBeamStiffnessFactor4
    ctBeamStiffnessFactor5 = St7API.ctBeamStiffnessFactor5
    ctBeamStiffnessFactor6 = St7API.ctBeamStiffnessFactor6
    ctBeamMassFactor = St7API.ctBeamMassFactor
    ctBeamSupportM1 = St7API.ctBeamSupportM1
    ctBeamSupportP1 = St7API.ctBeamSupportP1
    ctBeamSupportM2 = St7API.ctBeamSupportM2
    ctBeamSupportP2 = St7API.ctBeamSupportP2
    ctBeamSupportGapM1 = St7API.ctBeamSupportGapM1
    ctBeamSupportGapP1 = St7API.ctBeamSupportGapP1
    ctBeamSupportGapM2 = St7API.ctBeamSupportGapM2
    ctBeamSupportGapP2 = St7API.ctBeamSupportGapP2
    ctBeamTemperature = St7API.ctBeamTemperature
    ctBeamPreTension = St7API.ctBeamPreTension
    ctBeamPreStrain = St7API.ctBeamPreStrain
    ctBeamTempGradient1 = St7API.ctBeamTempGradient1
    ctBeamTempGradient2 = St7API.ctBeamTempGradient2
    ctBeamPipePressureIn = St7API.ctBeamPipePressureIn
    ctBeamPipePressureOut = St7API.ctBeamPipePressureOut
    ctBeamPipeTempIn = St7API.ctBeamPipeTempIn
    ctBeamPipeTempOut = St7API.ctBeamPipeTempOut
    ctBeamConvectionCoeff = St7API.ctBeamConvectionCoeff
    ctBeamConvectionAmbient = St7API.ctBeamConvectionAmbient
    ctBeamRadiationCoeff = St7API.ctBeamRadiationCoeff
    ctBeamRadiationAmbient = St7API.ctBeamRadiationAmbient
    ctBeamHeatFlux = St7API.ctBeamHeatFlux
    ctBeamHeatSource = St7API.ctBeamHeatSource
    ctBeamAgeAtFirstLoading = St7API.ctBeamAgeAtFirstLoading


class PlateContour(enum.Enum):
    ctPlateNone = St7API.ctPlateNone
    ctPlateAspectRatioMin = St7API.ctPlateAspectRatioMin
    ctPlateAspectRatioMax = St7API.ctPlateAspectRatioMax
    ctPlateWarping = St7API.ctPlateWarping
    ctPlateInternalAngle = St7API.ctPlateInternalAngle
    ctPlateInternalAngleRatio = St7API.ctPlateInternalAngleRatio
    ctPlateArea = St7API.ctPlateArea
    ctPlateAxis1 = St7API.ctPlateAxis1
    ctPlateAxis2 = St7API.ctPlateAxis2
    ctPlateAxis3 = St7API.ctPlateAxis3
    ctPlateDiscreteThicknessM = St7API.ctPlateDiscreteThicknessM
    ctPlateContinuousThicknessM = St7API.ctPlateContinuousThicknessM
    ctPlateDiscreteThicknessB = St7API.ctPlateDiscreteThicknessB
    ctPlateContinuousThicknessB = St7API.ctPlateContinuousThicknessB
    ctPlateOffset = St7API.ctPlateOffset
    ctPlateStiffnessFactor1 = St7API.ctPlateStiffnessFactor1
    ctPlateStiffnessFactor2 = St7API.ctPlateStiffnessFactor2
    ctPlateStiffnessFactor3 = St7API.ctPlateStiffnessFactor3
    ctPlateStiffnessFactor4 = St7API.ctPlateStiffnessFactor4
    ctPlateStiffnessFactor5 = St7API.ctPlateStiffnessFactor5
    ctPlateStiffnessFactor6 = St7API.ctPlateStiffnessFactor6
    ctPlateStiffnessFactor7 = St7API.ctPlateStiffnessFactor7
    ctPlateStiffnessFactor8 = St7API.ctPlateStiffnessFactor8
    ctPlateStiffnessFactor9 = St7API.ctPlateStiffnessFactor9
    ctPlateMassFactor = St7API.ctPlateMassFactor
    ctPlateEdgeNormalSupport = St7API.ctPlateEdgeNormalSupport
    ctPlateEdgeLateralSupport = St7API.ctPlateEdgeLateralSupport
    ctPlateEdgeSupportGap = St7API.ctPlateEdgeSupportGap
    ctPlateFaceNormalSupportMZ = St7API.ctPlateFaceNormalSupportMZ
    ctPlateFaceNormalSupportPZ = St7API.ctPlateFaceNormalSupportPZ
    ctPlateFaceLateralSupportMZ = St7API.ctPlateFaceLateralSupportMZ
    ctPlateFaceLateralSupportPZ = St7API.ctPlateFaceLateralSupportPZ
    ctPlateFaceSupportGapMZ = St7API.ctPlateFaceSupportGapMZ
    ctPlateFaceSupportGapPZ = St7API.ctPlateFaceSupportGapPZ
    ctPlateTemperature = St7API.ctPlateTemperature
    ctPlateTempGradient = St7API.ctPlateTempGradient
    ctPlatePreStressX = St7API.ctPlatePreStressX
    ctPlatePreStressY = St7API.ctPlatePreStressY
    ctPlatePreStressZ = St7API.ctPlatePreStressZ
    ctPlatePreStressMagnitude = St7API.ctPlatePreStressMagnitude
    ctPlatePreStrainX = St7API.ctPlatePreStrainX
    ctPlatePreStrainY = St7API.ctPlatePreStrainY
    ctPlatePreStrainZ = St7API.ctPlatePreStrainZ
    ctPlatePreStrainMagnitude = St7API.ctPlatePreStrainMagnitude
    ctPlatePreCurvatureX = St7API.ctPlatePreCurvatureX
    ctPlatePreCurvatureY = St7API.ctPlatePreCurvatureY
    ctPlatePreCurvatureMagnitude = St7API.ctPlatePreCurvatureMagnitude
    ctPlateEdgeNormalPressure = St7API.ctPlateEdgeNormalPressure
    ctPlateEdgeShear = St7API.ctPlateEdgeShear
    ctPlateEdgeTransverseShear = St7API.ctPlateEdgeTransverseShear
    ctPlateEdgeGlobalPressure = St7API.ctPlateEdgeGlobalPressure
    ctPlateEdgeGlobalPressureX = St7API.ctPlateEdgeGlobalPressureX
    ctPlateEdgeGlobalPressureY = St7API.ctPlateEdgeGlobalPressureY
    ctPlateEdgeGlobalPressureZ = St7API.ctPlateEdgeGlobalPressureZ
    ctPlatePressureNormalMZ = St7API.ctPlatePressureNormalMZ
    ctPlatePressureNormalPZ = St7API.ctPlatePressureNormalPZ
    ctPlatePressureGlobalMZ = St7API.ctPlatePressureGlobalMZ
    ctPlatePressureGlobalXMZ = St7API.ctPlatePressureGlobalXMZ
    ctPlatePressureGlobalYMZ = St7API.ctPlatePressureGlobalYMZ
    ctPlatePressureGlobalZMZ = St7API.ctPlatePressureGlobalZMZ
    ctPlatePressureGlobalPZ = St7API.ctPlatePressureGlobalPZ
    ctPlatePressureGlobalXPZ = St7API.ctPlatePressureGlobalXPZ
    ctPlatePressureGlobalYPZ = St7API.ctPlatePressureGlobalYPZ
    ctPlatePressureGlobalZPZ = St7API.ctPlatePressureGlobalZPZ
    ctPlateFaceShearX = St7API.ctPlateFaceShearX
    ctPlateFaceShearY = St7API.ctPlateFaceShearY
    ctPlateFaceShearMagnitude = St7API.ctPlateFaceShearMagnitude
    ctPlateNSMass = St7API.ctPlateNSMass
    ctPlateDynamicFactor = St7API.ctPlateDynamicFactor
    ctPlateConvectionCoeff = St7API.ctPlateConvectionCoeff
    ctPlateConvectionAmbient = St7API.ctPlateConvectionAmbient
    ctPlateRadiationCoeff = St7API.ctPlateRadiationCoeff
    ctPlateRadiationAmbient = St7API.ctPlateRadiationAmbient
    ctPlateHeatFlux = St7API.ctPlateHeatFlux
    ctPlateConvectionCoeffZPlus = St7API.ctPlateConvectionCoeffZPlus
    ctPlateConvectionCoeffZMinus = St7API.ctPlateConvectionCoeffZMinus
    ctPlateConvectionAmbientZPlus = St7API.ctPlateConvectionAmbientZPlus
    ctPlateConvectionAmbientZMinus = St7API.ctPlateConvectionAmbientZMinus
    ctPlateRadiationCoeffZPlus = St7API.ctPlateRadiationCoeffZPlus
    ctPlateRadiationCoeffZMinus = St7API.ctPlateRadiationCoeffZMinus
    ctPlateRadiationAmbientZPlus = St7API.ctPlateRadiationAmbientZPlus
    ctPlateRadiationAmbientZMinus = St7API.ctPlateRadiationAmbientZMinus
    ctPlateHeatSource = St7API.ctPlateHeatSource
    ctPlateSoilStressSV = St7API.ctPlateSoilStressSV
    ctPlateSoilStressK0 = St7API.ctPlateSoilStressK0
    ctPlateSoilStressSH = St7API.ctPlateSoilStressSH
    ctPlateSoilRatioOCR = St7API.ctPlateSoilRatioOCR
    ctPlateSoilRatioE0 = St7API.ctPlateSoilRatioE0
    ctPlateSoilFluidLevel = St7API.ctPlateSoilFluidLevel
    ctPlateAgeAtFirstLoading = St7API.ctPlateAgeAtFirstLoading


class BrickContour(enum.Enum):
    ctBrickNone = St7API.ctBrickNone
    ctBrickAspectRatioMin = St7API.ctBrickAspectRatioMin
    ctBrickAspectRatioMax = St7API.ctBrickAspectRatioMax
    ctBrickDeterminant = St7API.ctBrickDeterminant
    ctBrickInternalAngle = St7API.ctBrickInternalAngle
    ctBrickMixedProduct = St7API.ctBrickMixedProduct
    ctBrickDihedral = St7API.ctBrickDihedral
    ctBrickVolume = St7API.ctBrickVolume
    ctBrickAxis1 = St7API.ctBrickAxis1
    ctBrickAxis2 = St7API.ctBrickAxis2
    ctBrickAxis3 = St7API.ctBrickAxis3
    ctBrickNormalSupport = St7API.ctBrickNormalSupport
    ctBrickLateralSupport = St7API.ctBrickLateralSupport
    ctBrickSupportGap = St7API.ctBrickSupportGap
    ctBrickTemperature = St7API.ctBrickTemperature
    ctBrickPreStressX = St7API.ctBrickPreStressX
    ctBrickPreStressY = St7API.ctBrickPreStressY
    ctBrickPreStressZ = St7API.ctBrickPreStressZ
    ctBrickPreStressMagnitude = St7API.ctBrickPreStressMagnitude
    ctBrickPreStrainX = St7API.ctBrickPreStrainX
    ctBrickPreStrainY = St7API.ctBrickPreStrainY
    ctBrickPreStrainZ = St7API.ctBrickPreStrainZ
    ctBrickPreStrainMagnitude = St7API.ctBrickPreStrainMagnitude
    ctBrickNormalPressure = St7API.ctBrickNormalPressure
    ctBrickGlobalPressure = St7API.ctBrickGlobalPressure
    ctBrickGlobalPressureX = St7API.ctBrickGlobalPressureX
    ctBrickGlobalPressureY = St7API.ctBrickGlobalPressureY
    ctBrickGlobalPressureZ = St7API.ctBrickGlobalPressureZ
    ctBrickShearX = St7API.ctBrickShearX
    ctBrickShearY = St7API.ctBrickShearY
    ctBrickShearMagnitude = St7API.ctBrickShearMagnitude
    ctBrickNSMass = St7API.ctBrickNSMass
    ctBrickDynamicFactor = St7API.ctBrickDynamicFactor
    ctBrickConvectionCoeff = St7API.ctBrickConvectionCoeff
    ctBrickConvectionAmbient = St7API.ctBrickConvectionAmbient
    ctBrickRadiationCoeff = St7API.ctBrickRadiationCoeff
    ctBrickRadiationAmbient = St7API.ctBrickRadiationAmbient
    ctBrickHeatFlux = St7API.ctBrickHeatFlux
    ctBrickHeatSource = St7API.ctBrickHeatSource
    ctBrickSoilStressSV = St7API.ctBrickSoilStressSV
    ctBrickSoilStressK0 = St7API.ctBrickSoilStressK0
    ctBrickSoilStressSH = St7API.ctBrickSoilStressSH
    ctBrickSoilRatioOCR = St7API.ctBrickSoilRatioOCR
    ctBrickSoilRatioE0 = St7API.ctBrickSoilRatioE0
    ctBrickSoilFluidLevel = St7API.ctBrickSoilFluidLevel
    ctBrickAgeAtFirstLoading = St7API.ctBrickAgeAtFirstLoading


class ImageType(enum.Enum):
    itBitmap8Bit = St7API.itBitmap8Bit
    itBitmap16Bit = St7API.itBitmap16Bit
    itBitmap24Bit = St7API.itBitmap24Bit
    itJPEG = St7API.itJPEG
    itPNG = St7API.itPNG


class ScaleType(enum.Enum):
    dsPercent = St7API.dsPercent
    dsAbsolute = St7API.dsAbsolute


class St7Model:
    _fn: str = None
    _temp_dir: str = None
    uID: int = 1

    def __init__(self, fn_st7: T_Path, temp_dir=None):
        self._fn = str(fn_st7)

        if temp_dir:
            self._temp_dir = str(temp_dir)
        else:
            self._temp_dir = tempfile.gettempdir()

        chk(St7API.St7Init())
        chk(St7API.St7OpenFile(self.uID, self._fn.encode(), self._temp_dir.encode()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        chk(St7API.St7CloseFile(self.uID))
        chk(St7API.St7Release())

    def open_results(self, fn_res: T_Path) -> "St7Results":
        return St7Results(self, fn_res)

    def entity_numbers(self, entity: Entity) -> range:
        ct_max_num = ctypes.c_long()
        chk(St7API.St7GetTotal(self.uID, entity.value, ct_max_num))
        return range(1, ct_max_num.value+1)

    def St7NewLoadCase(self, case_name: str):
        chk(St7API.St7NewLoadCase(self.uID, case_name.encode()))

    def St7GetNumLoadCase(self) -> int:
        ct_num_cases = ctypes.c_long()
        chk(St7API.St7GetNumLoadCase(self.uID, ct_num_cases))
        return ct_num_cases.value

    def St7SetPlatePreLoad3(self, iPlate: int, iLoadCase: int, load_type: PreLoadType, load: Vector3):
        doubles = (ctypes.c_double * 3)(*load)
        chk(St7API.St7SetPlatePreLoad3(
            self.uID,
            iPlate,
            iLoadCase,
            load_type.value,
            doubles
        ))

    def St7RunSolver(self, solver: SolverType, solver_mode: SolverMode, wait: bool):
        chk(St7API.St7RunSolver(
            self.uID,
            solver.value,
            solver_mode.value,
            wait
        ))

    def St7EnableNLALoadCase(self, stage: int, load_case_num: int):
        chk(St7API.St7EnableNLALoadCase(
            self.uID,
            stage,
            load_case_num,
        ))

    def St7DisableNLALoadCase(self, stage: int, load_case_num: int):
        chk(St7API.St7DisableNLALoadCase(
            self.uID,
            stage,
            load_case_num,
        ))

    def St7EnableNLAFreedomCase(self, stage: int, freedom_case_num: int):
        chk(St7API.St7EnableNLAFreedomCase(
            self.uID,
            stage,
            freedom_case_num,
        ))


    def St7AddNLAIncrement(self, stage: int, inc_name: str):
        chk(St7API.St7AddNLAIncrement(
            self.uID,
            stage,
            inc_name.encode()
        ))

    def St7SetNLALoadIncrementFactor(self, stage: int, increment: int, load_case_num: int, factor: float):
        chk(St7API.St7SetNLALoadIncrementFactor(
            self.uID,
            stage,
            increment,
            load_case_num,
            factor
        ))

    def St7SetNLAFreedomIncrementFactor(self, stage: int, increment: int, freedom_case_num: int, factor: float):
        chk(St7API.St7SetNLAFreedomIncrementFactor(
            self.uID,
            stage,
            increment,
            freedom_case_num,
            factor
        ))

    def St7GetNumNLAIncrements(self, stage: int) -> int:
        ct_incs = ctypes.c_long()
        chk(St7API.St7GetNumNLAIncrements(self.uID, stage, ct_incs))
        return ct_incs.value

    def St7EnableSaveRestart(self):
        chk(St7API.St7EnableSaveRestart(self.uID))

    def St7EnableSaveLastRestartStep(self):
        chk(St7API.St7EnableSaveLastRestartStep(self.uID))

    def St7SetNLAInitial(self, fn_res: T_Path, case_num: int):
        chk(St7API.St7SetNLAInitial(
            self.uID,
            str(fn_res).encode(),
            case_num
        ))

    def St7SetQSAInitial(self, fn_res: T_Path, case_num: int):
        chk(St7API.St7SetQSAInitial(
            self.uID,
            str(fn_res).encode(),
            case_num
        ))

    def St7SetResultFileName(self, fn_res: T_Path):
        chk(St7API.St7SetResultFileName(
            self.uID,
            str(fn_res).encode(),
        ))

    def St7SetStaticRestartFile(self, fn_restart: T_Path):
        chk(St7API.St7SetStaticRestartFile(
            self.uID,
            str(fn_restart).encode(),
        ))

    def St7EnableTransientLoadCase(self, case_num: int):
        chk(St7API.St7EnableTransientLoadCase(self.uID, case_num))

    def St7DisableTransientLoadCase(self, case_num: int):
        chk(St7API.St7DisableTransientLoadCase(self.uID, case_num))

    def St7EnableTransientFreedomCase(self, case_num: int):
        chk(St7API.St7EnableTransientFreedomCase(self.uID, case_num))

    def St7SetTransientLoadTimeTable(self, case_num: int, table_num: int, add_time_steps: bool):
        chk(St7API.St7SetTransientLoadTimeTable(self.uID, case_num, table_num, add_time_steps))

    def St7SetTransientFreedomTimeTable(self, case_num: int, table_num: int, add_time_steps: bool):
        chk(St7API.St7SetTransientFreedomTimeTable(self.uID, case_num, table_num, add_time_steps))

    def St7SaveFile(self):
        chk(St7API.St7SaveFile(self.uID))

    def St7SaveFileCopy(self, fn_st7: str):
        chk(St7API.St7SaveFileCopy(self.uID, str(fn_st7).encode()))

    def St7GetElementCentroid(self, entity: Entity, elem_num: int, face_edge_num: int) -> Vector3:
        ct_xyz = (ctypes.c_double * 3)()
        chk(St7API.St7GetElementCentroid(self.uID, entity.value, elem_num, face_edge_num, ct_xyz))
        return Vector3(*ct_xyz)

    def St7GetElementData(self, entity: Entity, elem_num: int) -> float:
        ct_data = ctypes.c_double()
        chk(St7API.St7GetElementData(self.uID, entity.value, elem_num, ct_data))
        return ct_data.value

    def St7GetNodeXYZ(self, node_num: int) -> Vector3:
        ct_xyz = (ctypes.c_double * 3)()
        chk(St7API.St7GetNodeXYZ(self.uID, node_num, ct_xyz))
        return Vector3(*ct_xyz)

    def St7GetElementConnection(self, entity: Entity, elem_num: int) -> typing.Tuple[int, ...]:
        ct_conn = (ctypes.c_long * 30)()
        chk(St7API.St7GetElementConnection(self.uID, entity.value, elem_num, ct_conn))
        n = ct_conn[0]
        return tuple(ct_conn[1: 1+n])

    def St7CreateModelWindow(self, dont_really_make: bool) -> "St7ModelWindow":
        if dont_really_make:
            return St7ModelWindowDummy(model=self)

        else:
            return St7ModelWindow(model=self)

    def St7SetUseSolverDLL(self, use_dll: bool):
        ct_int = ctypes.c_long(use_dll)
        chk(St7API.St7SetUseSolverDLL(ct_int))


    def St7SetNumTimeStepRows(self, num_rows: int):
        chk(St7API.St7SetNumTimeStepRows(self.uID, num_rows))

    def St7SetTimeStepData(self, row: int, num_steps: int, save_every: int, time_step: float):
        chk(St7API.St7SetTimeStepData(self.uID, row, num_steps, save_every, time_step))

    def St7SetSolverDefaultsLogical(self, solver_def_logical: SolverDefaultLogical, value: bool):
        chk(St7API.St7SetSolverDefaultsLogical(self.uID, solver_def_logical.value, value))

    def St7SetSolverDefaultsInteger(self, solver_def_int: SolverDefaultInteger, value: int):
        chk(St7API.St7SetSolverDefaultsInteger(self.uID, solver_def_int.value, value))


    def _make_table_data_and_validate(self, num_entries: int, doubles: typing.Sequence[float]):
        data = list(doubles)
        if len(data) != num_entries*2:
            raise ValueError("Mismatch!")

        ct_doubles_type = (ctypes.c_double * (2 * num_entries))
        ct_doubles = ct_doubles_type(*data)
        return ct_doubles

    def St7NewTableType(self, table_type: TableType, table_id: int, num_entries: int, table_name: str, doubles: typing.Sequence[float]):
        ct_doubles = self._make_table_data_and_validate(num_entries, doubles)
        chk(St7API.St7NewTableType(self.uID, table_type.value, table_id, num_entries, table_name.encode(), ct_doubles))

    def St7SetTableTypeData(self, table_type: TableType, table_id: int, num_entries: int, doubles: typing.Sequence[float]):
        ct_doubles = self._make_table_data_and_validate(num_entries, doubles)
        chk(St7API.St7SetTableTypeData(self.uID, table_type.value, table_id, num_entries, ct_doubles))

    def St7SetPlateXAngle1(self, plate_num: int, ang_deg: float):
        ct_doubles = (ctypes.c_double * 10)()
        ct_doubles[0] = ang_deg

        chk(St7API.St7SetPlateXAngle1(self.uID, plate_num, ct_doubles))

    def St7GetPlateXAngle1(self, plate_num: int) -> float:
        ct_doubles = (ctypes.c_double * 10)()

        chk(St7API.St7GetPlateXAngle1(self.uID, plate_num, ct_doubles))
        return ct_doubles[0]



class St7Results:
    model: St7Model = None
    fn_res: str = None
    uID: int = None
    primary_cases: range = None

    def __init__(self, model: St7Model, fn_res: T_Path):

        self.model = model
        self.fn_res = str(fn_res)
        self.uID = self.model.uID

        ct_num_prim, ct_num_sec = ctypes.c_long(), ctypes.c_long()
        chk(St7API.St7OpenResultFile(self.uID, self.fn_res.encode(), b'', False, ct_num_prim, ct_num_sec))

        self.primary_cases = range(1, ct_num_prim.value+1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        chk(St7API.St7CloseResultFile(self.uID))

    def St7GetNodeResult(
            self,
            res_type: NodeResultType,
            node_num: int,
            res_case: int,
    ) -> ResultOutput:
        ct_res_array = (ctypes.c_double * 6)()

        chk(St7API.St7GetNodeResult(
            self.uID,
            res_type.value,
            node_num,
            res_case,
            ct_res_array))

        out_array = tuple(ct_res_array)

        return ResultOutput(
            num_points=1,
            num_cols=6,
            results=out_array,
        )


    def St7GetPlateResultArray(
            self,
            res_type: PlateResultType,
            res_sub_type: typing.Union[PlateResultSubType, int],
            plate_num: int,
            case_num: int,
            sample_location: SampleLocation,
            surface: PlateSurface,
            layer: int,
            ) -> ResultOutput:

        if isinstance(res_sub_type, PlateResultSubType):
            real_sub_type = res_sub_type.value

        elif isinstance(res_sub_type, int):
            real_sub_type = res_sub_type

        else:
            raise TypeError(res_sub_type)

        ct_res_array = (ctypes.c_double * St7API.kMaxPlateResult)()
        ct_num_points = ctypes.c_long()
        ct_num_cols = ctypes.c_long()

        chk(St7API.St7GetPlateResultArray(
            self.uID,
            res_type.value,
            real_sub_type,
            plate_num,
            case_num,
            sample_location.value,
            surface.value,
            layer,
            ct_num_points,
            ct_num_cols,
            ct_res_array
        ))

        out_array = tuple(ct_res_array[0:ct_num_points.value * ct_num_cols.value])
        return ResultOutput(
            num_points=ct_num_points.value,
            num_cols=ct_num_cols.value,
            results=out_array,
        )


class St7ModelWindow:
    model: St7Model = None
    uID: int = None

    def __init__(self, model: St7Model):
        self.model = model
        self.uID = self.model.uID
        chk(St7API.St7CreateModelWindow(self.uID))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't check for an error here.
        St7API.St7DestroyModelWindow(self.uID)

    def close(self):
        chk(St7API.St7DestroyModelWindow(self.uID))

    def St7ShowModelWindow(self):
        chk(St7API.St7ShowModelWindow(self.uID))

    def St7DestroyModelWindow(self):
        chk(St7API.St7DestroyModelWindow(self.uID))

    def St7GetDrawAreaSize(self) -> CanvasSize:
        ct_width, ct_height = ctypes.c_long(), ctypes.c_long()
        chk(St7API.St7GetDrawAreaSize(self.uID, ct_width, ct_height))
        return CanvasSize(width=ct_width.value, height=ct_height.value)

    def St7PositionModelWindow(self, left: int, top: int, width: int, height: int):
        chk(St7API.St7PositionModelWindow(self.uID, left, top, width, height))

    def St7SetEntityContourIndex(self, entity: Entity, index: typing.Union[BeamContour, PlateContour, BrickContour]):
        chk(St7API.St7SetEntityContourIndex(self.uID, entity.value, index.value))

    def St7ExportImage(self, fn: T_Path, image_type: ImageType, width: int, height: int):
        chk(St7API.St7ExportImage(self.uID, str(fn).encode(), image_type.value, width, height))

    def St7SetPlateResultDisplay_None(self):
        integers = [0] * 15
        integers[St7API.ipResultType] = St7API.rtAsNone
        self.St7SetPlateResultDisplay(integers)

    def St7SetPlateResultDisplay(self, integers: typing.Tuple[int]):
        ct_ints = (ctypes.c_long*20)()
        ct_ints[:len(integers)] = integers[:]

    def St7SetWindowResultCase(self, case_num: int):
        chk(St7API.St7SetWindowResultCase(self.uID, case_num))

    def St7RedrawModel(self, rescale: bool):
        chk(St7API.St7RedrawModel(self.uID, rescale))

    def St7SetDisplacementScale(self, disp_scale: float, scale_type: ScaleType):
        chk(St7API.St7SetDisplacementScale(self.uID, disp_scale, scale_type.value))

    def St7GetEntityContourSettingsStyle(self, entity: Entity) -> ContourSettingsStyle:
        ints = ContourSettingsStyle.get_single_sub_array_of_type(ctypes.c_long)
        ints_arr = ints.make_empty_array()

        chk(St7API.St7GetEntityContourSettingsStyle(self.uID, entity.value, ints_arr))

        ints_instance = ints.instance_from_st7_array(ints_arr)
        return ContourSettingsStyle.instance_from_sub_array_instances(ints_instance)


    def St7SetEntityContourSettingsStyle(self, entity: Entity, contour_settings_style: ContourSettingsStyle):
        ints_arr = contour_settings_style.get_single_sub_array_instance_of_type(ctypes.c_long).to_st7_array()
        chk(St7API.St7SetEntityContourSettingsStyle(self.uID, entity.value, ints_arr))


    def St7GetEntityContourSettingsLimits(self, entity: Entity) -> ContourSettingsLimit:
        ints = ContourSettingsLimit.get_single_sub_array_of_type(ctypes.c_long)
        doubles = ContourSettingsLimit.get_single_sub_array_of_type(ctypes.c_double)

        ints_arr = ints.make_empty_array()
        doubles_arr = doubles.make_empty_array()

        chk(St7API.St7GetEntityContourSettingsLimits(self.uID, entity.value, ints_arr, doubles_arr))

        ints_instance = ints.instance_from_st7_array(ints_arr)
        doubles_instance = doubles.instance_from_st7_array(doubles_arr)

        contour_settings_limit = ContourSettingsLimit.instance_from_sub_array_instances(ints_instance, doubles_instance)

        return contour_settings_limit


    def St7SetEntityContourSettingsLimits(self, entity: Entity, contour_settings_limit: ContourSettingsLimit):
        
        ints_arr = contour_settings_limit.get_single_sub_array_instance_of_type(ctypes.c_long).to_st7_array()
        doubles_arr = contour_settings_limit.get_single_sub_array_instance_of_type(ctypes.c_double).to_st7_array()

        chk(St7API.St7SetEntityContourSettingsLimits(self.uID, entity.value, ints_arr, doubles_arr))




def _DummyClassFactory(name, BaseClass):
    """Utility function to make a class with no-op methods for everything, but the same signature."""
    def make_no_op_function(func_returns_self: bool) -> typing.Callable:
        # Factory to make
        if func_returns_self:
            def f(self, *args, **kwargs):
                return self

        else:
            def f(*args, **kwargs):
                pass

        return f

    # Build a new dictionary of no-op functions for all the user-defined things in the base class.
    attribute_dict = {}
    for attr_name in dir(BaseClass):
        attr = getattr(BaseClass, attr_name)

        is_dunder = attr_name.startswith("__")
        is_callable = callable(attr)
        if not is_dunder:
            if is_callable:
                # For methods
                attribute_dict[attr_name] = make_no_op_function(func_returns_self=False)

            else:
                # For other attributes
                attribute_dict[attr_name] = None

    # Add the special cases.
    attribute_dict["__init__"] = make_no_op_function(func_returns_self=False)
    attribute_dict["__enter__"] = make_no_op_function(func_returns_self=True)
    attribute_dict["__exit__"] = make_no_op_function(func_returns_self=False)

    # The rest can be inherited.
    NewClass = type(name, (BaseClass,), attribute_dict)
    return NewClass


St7ModelWindowDummy = _DummyClassFactory("St7ModelWindowDummy", St7ModelWindow)
