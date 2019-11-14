# Friendly wrapper around the Strand7 API.

import St7API
import tempfile
import enum
import typing
import ctypes

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
    stLinearStaticSolver = St7API.stLinearStaticSolver
    stLinearBucklingSolver = St7API.stLinearBucklingSolver
    stNonlinearStaticSolver = St7API.stNonlinearStaticSolver
    stNaturalFrequencySolver = St7API.stNaturalFrequencySolver
    stHarmonicResponseSolver = St7API.stHarmonicResponseSolver
    stSpectralResponseSolver = St7API.stSpectralResponseSolver
    stLinearTransientDynamicSolver = St7API.stLinearTransientDynamicSolver
    stNonlinearTransientDynamicSolver = St7API.stNonlinearTransientDynamicSolver
    stSteadyHeatSolver = St7API.stSteadyHeatSolver
    stTransientHeatSolver = St7API.stTransientHeatSolver
    stLoadInfluenceSolver = St7API.stLoadInfluenceSolver
    stQuasiStaticSolver = St7API.stQuasiStaticSolver


class SolverMode(enum.Enum):
    smNormalRun = St7API.smNormalRun
    smNormalCloseRun = St7API.smNormalCloseRun
    smProgressRun = St7API.smProgressRun
    smBackgroundRun = St7API.smBackgroundRun


class PreLoadType(enum.Enum):
    plPlatePreStrain = St7API.plPlatePreStrain
    plPlatePreStress = St7API.plPlatePreStress


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
    ctPlateDiscreteThicknessM = St7API.ctPlateDiscreteThicknessM
    ctPlateContinuousThicknessM = St7API.ctPlateContinuousThicknessM
    ctPlateDiscreteThicknessB = St7API.ctPlateDiscreteThicknessB
    ctPlateContinuousThicknessB = St7API.ctPlateContinuousThicknessB
    ctPlateOffset = St7API.ctPlateOffset
    ctPlateArea = St7API.ctPlateArea
    ctPlateAxis1 = St7API.ctPlateAxis1
    ctPlateAxis2 = St7API.ctPlateAxis2
    ctPlateAxis3 = St7API.ctPlateAxis3
    ctPlateTemperature = St7API.ctPlateTemperature
    ctPlateEdgeNormalSupport = St7API.ctPlateEdgeNormalSupport
    ctPlateEdgeLateralSupport = St7API.ctPlateEdgeLateralSupport
    ctPlateEdgeSupportGap = St7API.ctPlateEdgeSupportGap
    ctPlateFaceNormalSupportMZ = St7API.ctPlateFaceNormalSupportMZ
    ctPlateFaceLateralSupportMZ = St7API.ctPlateFaceLateralSupportMZ
    ctPlateFaceNormalSupportPZ = St7API.ctPlateFaceNormalSupportPZ
    ctPlateFaceLateralSupportPZ = St7API.ctPlateFaceLateralSupportPZ
    ctPlateFaceSupportGapMZ = St7API.ctPlateFaceSupportGapMZ
    ctPlateFaceSupportGapPZ = St7API.ctPlateFaceSupportGapPZ
    ctPlatePreStressX = St7API.ctPlatePreStressX
    ctPlatePreStressY = St7API.ctPlatePreStressY
    ctPlatePreStressZ = St7API.ctPlatePreStressZ
    ctPlatePreStressMagnitude = St7API.ctPlatePreStressMagnitude
    ctPlatePreStrainX = St7API.ctPlatePreStrainX
    ctPlatePreStrainY = St7API.ctPlatePreStrainY
    ctPlatePreStrainZ = St7API.ctPlatePreStrainZ
    ctPlatePreStrainMagnitude = St7API.ctPlatePreStrainMagnitude
    ctPlateTempGradient = St7API.ctPlateTempGradient
    ctPlateEdgePressure = St7API.ctPlateEdgePressure
    ctPlateEdgeShear = St7API.ctPlateEdgeShear
    ctPlateEdgeNormalShear = St7API.ctPlateEdgeNormalShear
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
    ctPlateSoilStressKO = St7API.ctPlateSoilStressKO
    ctPlateSoilStressSH = St7API.ctPlateSoilStressSH
    ctPlateSoilRatioOCR = St7API.ctPlateSoilRatioOCR
    ctPlateSoilRatioEO = St7API.ctPlateSoilRatioEO
    ctPlateSoilFluidLevel = St7API.ctPlateSoilFluidLevel
    ctPlateAgeAtFirstLoading = St7API.ctPlateAgeAtFirstLoading


class BrickContour(enum.Enum):
    ctBrickNone = St7API.ctBrickNone
    ctBrickAspectRatioMin = St7API.ctBrickAspectRatioMin
    ctBrickAspectRatioMax = St7API.ctBrickAspectRatioMax
    ctBrickVolume = St7API.ctBrickVolume
    ctBrickDeterminant = St7API.ctBrickDeterminant
    ctBrickInternalAngle = St7API.ctBrickInternalAngle
    ctBrickMixedProduct = St7API.ctBrickMixedProduct
    ctBrickDihedral = St7API.ctBrickDihedral
    ctBrickAxis1 = St7API.ctBrickAxis1
    ctBrickAxis2 = St7API.ctBrickAxis2
    ctBrickAxis3 = St7API.ctBrickAxis3
    ctBrickTemperature = St7API.ctBrickTemperature
    ctBrickNormalSupport = St7API.ctBrickNormalSupport
    ctBrickLateralSupport = St7API.ctBrickLateralSupport
    ctBrickSupportGap = St7API.ctBrickSupportGap
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
    ctBrickSoilStressKO = St7API.ctBrickSoilStressKO
    ctBrickSoilStressSH = St7API.ctBrickSoilStressSH
    ctBrickSoilRatioOCR = St7API.ctBrickSoilRatioOCR
    ctBrickSoilRatioEO = St7API.ctBrickSoilRatioEO
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

    def __init__(self, fn_st7: str, temp_dir=None):
        self._fn = str(fn_st7)

        if temp_dir:
            self._temp_dir = temp_dir
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

    def open_results(self, fn_res: str) -> "St7Results":
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

    def St7SetNLAInitial(self, fn_restart: str, case_num: int):
        chk(St7API.St7SetNLAInitial(
            self.uID,
            str(fn_restart).encode(),
            case_num
        ))

    def St7SetResultFileName(self, fn_res: str):
        chk(St7API.St7SetResultFileName(
            self.uID,
            str(fn_res).encode(),
        ))

    def St7SetStaticRestartFile(self, fn_restart: str):
        chk(St7API.St7SetStaticRestartFile(
            self.uID,
            str(fn_restart).encode(),
        ))

    def St7SaveFile(self):
        chk(St7API.St7SaveFile(self.uID))

    def St7SaveFileCopy(self, fn_st7: str):
        chk(St7API.St7SaveFileCopy(self.uID, str(fn_st7).encode()))

    def St7GetElementCentroid(self, entity: Entity, elem_num: int, face_edge_num: int) -> Vector3:
        ct_xyz = (ctypes.c_double * 3)()
        chk(St7API.St7GetElementCentroid(self.uID, entity.value, elem_num, face_edge_num, ct_xyz))
        return Vector3(*ct_xyz)

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


class St7Results:
    model: St7Model = None
    fn_res: str = None
    uID: int = None
    primary_cases: range = None

    def __init__(self, model: St7Model, fn_res: str):

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


    def St7SetDisplacementScale(self, disp_scale: float, scale_type: ScaleType):
        chk(St7API.St7SetDisplacementScale(self.uID, disp_scale, scale_type.value))


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

    def St7ExportImage(self, fn: str, image_type: ImageType, width: int, height: int):
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
