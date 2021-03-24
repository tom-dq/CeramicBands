# Some handy bits and pieces

import shutil
from PIL import Image
import os
import time
import multiprocessing
import typing

BASE_DIR = r"C:\Simulations\CeramicBandsData\LocalTest\v7-Wedge\pics"

def tally():
    for base, dirs, files in os.walk(BASE_DIR):
        png_files = [os.path.join(base, f) for f in files if os.path.splitext(f)[1].lower() == '.png']
        num_png = len(png_files)
        if num_png > 10:
            print(num_png, base, sep='\t')


class Job(typing.NamedTuple):
    idx: int
    fn: str

def compress_png(job):
    png_fn = job.fn
    if job.idx % 100 == 0:
        print(job)

    image = Image.open(png_fn)
    image.save(png_fn, optimize=True, quality=95)


def compress_existing_images():
    def all_pngs():
        for base, dirs, files in os.walk(BASE_DIR):
            png_files = (os.path.join(base, f) for f in files if os.path.splitext(f)[1].lower() == '.png')
            yield from png_files

    def convertible_png(fn):
        stat = os.stat(fn)
        size_mb = stat.st_size / 1024 / 1024
        age_in_mins = (time.time() - stat.st_mtime) / 60

        return size_mb > 1.0 and age_in_mins > 10.0

    conv_pngs = (fn for fn in all_pngs() if convertible_png(fn))

    jobs = (Job(idx=idx, fn=fn) for idx, fn in enumerate(conv_pngs))

    with multiprocessing.Pool(6) as pool:
        res = pool.map_async(compress_png, jobs)
        pool.close()
        pool.join()


def extract_final_images():
    final_image_dir = os.path.join(BASE_DIR, "_LastPics")

    def dirs_and_last_pngs():
        for base, dirs, files in os.walk(BASE_DIR):
            png_files = list(os.path.join(base, f) for f in files if os.path.splitext(f)[1].lower() == '.png')
            if len(png_files) > 5:
                png_files.sort()
                yield base, png_files[-1]

    for dir_name, one_png in dirs_and_last_pngs():
        just_dir = os.path.split(dir_name)[1]
        just_fn = os.path.split(one_png)[1]
        out_name = f"{just_dir}_{just_fn}"

        out_fn = os.path.join(final_image_dir, out_name)
        shutil.copy2(one_png, out_fn)


def make_enum(text):
    data = [one_line.split("=")[0].strip() for one_line in text.splitlines() if "=" in one_line]
    print("class BLAH_BLAH_BLAH(enum.Enum):")
    for x in data:
        print(f"    {x} = St7API.{x}")

    print("")

if __name__ == "__main__":
    text = """ctBeamNone = 0
ctBeamLength = 1
ctBeamAxis1 = 2
ctBeamAxis2 = 3
ctBeamAxis3 = 4
ctBeamEA = 5
ctBeamEI11 = 6
ctBeamEI22 = 7
ctBeamGJ = 8
ctBeamEAFactor = 9
ctBeamEI11Factor = 10
ctBeamEI22Factor = 11
ctBeamGJFactor = 12
ctBeamOffset1 = 13
ctBeamOffset2 = 14
ctBeamStiffnessFactor1 = 15
ctBeamStiffnessFactor2 = 16
ctBeamStiffnessFactor3 = 17
ctBeamStiffnessFactor4 = 18
ctBeamStiffnessFactor5 = 19
ctBeamStiffnessFactor6 = 20
ctBeamMassFactor = 21
ctBeamSupportM1 = 22
ctBeamSupportP1 = 23
ctBeamSupportM2 = 24
ctBeamSupportP2 = 25
ctBeamSupportGapM1 = 26
ctBeamSupportGapP1 = 27
ctBeamSupportGapM2 = 28
ctBeamSupportGapP2 = 29
ctBeamTemperature = 30
ctBeamPreTension = 31
ctBeamPreStrain = 32
ctBeamTempGradient1 = 33
ctBeamTempGradient2 = 34
ctBeamPipePressureIn = 35
ctBeamPipePressureOut = 36
ctBeamPipeTempIn = 37
ctBeamPipeTempOut = 38
ctBeamConvectionCoeff = 39
ctBeamConvectionAmbient = 40
ctBeamRadiationCoeff = 41
ctBeamRadiationAmbient = 42
ctBeamHeatFlux = 43
ctBeamHeatSource = 44
ctBeamAgeAtFirstLoading = 45

## Entity Display Settings - Plate Contour Types
ctPlateNone = 0
ctPlateAspectRatioMin = 1
ctPlateAspectRatioMax = 2
ctPlateWarping = 3
ctPlateInternalAngle = 4
ctPlateInternalAngleRatio = 5
ctPlateDiscreteThicknessM = 6
ctPlateContinuousThicknessM = 7
ctPlateDiscreteThicknessB = 8
ctPlateContinuousThicknessB = 9
ctPlateOffset = 10
ctPlateArea = 11
ctPlateAxis1 = 12
ctPlateAxis2 = 13
ctPlateAxis3 = 14
ctPlateTemperature = 15
ctPlateEdgeNormalSupport = 16
ctPlateEdgeLateralSupport = 17
ctPlateEdgeSupportGap = 18
ctPlateFaceNormalSupportMZ = 19
ctPlateFaceLateralSupportMZ = 20
ctPlateFaceNormalSupportPZ = 21
ctPlateFaceLateralSupportPZ = 22
ctPlateFaceSupportGapMZ = 23
ctPlateFaceSupportGapPZ = 24
ctPlatePreStressX = 25
ctPlatePreStressY = 26
ctPlatePreStressZ = 27
ctPlatePreStressMagnitude = 28
ctPlatePreStrainX = 29
ctPlatePreStrainY = 30
ctPlatePreStrainZ = 31
ctPlatePreStrainMagnitude = 32
ctPlateTempGradient = 33
ctPlateEdgePressure = 34
ctPlateEdgeShear = 35
ctPlateEdgeNormalShear = 36
ctPlatePressureNormalMZ = 37
ctPlatePressureNormalPZ = 38
ctPlatePressureGlobalMZ = 39
ctPlatePressureGlobalXMZ = 40
ctPlatePressureGlobalYMZ = 41
ctPlatePressureGlobalZMZ = 42
ctPlatePressureGlobalPZ = 43
ctPlatePressureGlobalXPZ = 44
ctPlatePressureGlobalYPZ = 45
ctPlatePressureGlobalZPZ = 46
ctPlateFaceShearX = 47
ctPlateFaceShearY = 48
ctPlateFaceShearMagnitude = 49
ctPlateNSMass = 50
ctPlateDynamicFactor = 51
ctPlateConvectionCoeff = 52
ctPlateConvectionAmbient = 53
ctPlateRadiationCoeff = 54
ctPlateRadiationAmbient = 55
ctPlateHeatFlux = 56
ctPlateConvectionCoeffZPlus = 57
ctPlateConvectionCoeffZMinus = 58
ctPlateConvectionAmbientZPlus = 59
ctPlateConvectionAmbientZMinus = 60
ctPlateRadiationCoeffZPlus = 61
ctPlateRadiationCoeffZMinus = 62
ctPlateRadiationAmbientZPlus = 63
ctPlateRadiationAmbientZMinus = 64
ctPlateHeatSource = 65
ctPlateSoilStressSV = 66
ctPlateSoilStressKO = 67
ctPlateSoilStressSH = 68
ctPlateSoilRatioOCR = 69
ctPlateSoilRatioEO = 70
ctPlateSoilFluidLevel = 71
ctPlateAgeAtFirstLoading = 72

## Entity Display Settings - Brick Contour Types
ctBrickNone = 0
ctBrickAspectRatioMin = 1
ctBrickAspectRatioMax = 2
ctBrickVolume = 3
ctBrickDeterminant = 4
ctBrickInternalAngle = 5
ctBrickMixedProduct = 6
ctBrickDihedral = 7
ctBrickAxis1 = 8
ctBrickAxis2 = 9
ctBrickAxis3 = 10
ctBrickTemperature = 11
ctBrickNormalSupport = 12
ctBrickLateralSupport = 13
ctBrickSupportGap = 14
ctBrickPreStressX = 15
ctBrickPreStressY = 16
ctBrickPreStressZ = 17
ctBrickPreStressMagnitude = 18
ctBrickPreStrainX = 19
ctBrickPreStrainY = 20
ctBrickPreStrainZ = 21
ctBrickPreStrainMagnitude = 22
ctBrickNormalPressure = 23
ctBrickGlobalPressure = 24
ctBrickGlobalPressureX = 25
ctBrickGlobalPressureY = 26
ctBrickGlobalPressureZ = 27
ctBrickShearX = 28
ctBrickShearY = 29
ctBrickShearMagnitude = 30
ctBrickNSMass = 31
ctBrickDynamicFactor = 32
ctBrickConvectionCoeff = 33
ctBrickConvectionAmbient = 34
ctBrickRadiationCoeff = 35
ctBrickRadiationAmbient = 36
ctBrickHeatFlux = 37
ctBrickHeatSource = 38
ctBrickSoilStressSV = 39
ctBrickSoilStressKO = 40
ctBrickSoilStressSH = 41
ctBrickSoilRatioOCR = 42
ctBrickSoilRatioEO = 43
ctBrickSoilFluidLevel = 44
ctBrickAgeAtFirstLoading = 45"""

    text2 = """tyNODE = 0
tyBEAM = 1
tyPLATE = 2
tyBRICK = 3
tyLINK = 4
tyVERTEX = 5
tyGEOMETRYEDGE = 6
tyGEOMETRYFACE = 7
tyLOADPATH = 8
tyGEOMETRYCOEDGE = 9
tyGEOMETRYLOOP = 10"""


    text3 = """rtPlateStress = 1
rtPlateStrain = 2
rtPlateEnergy = 3
rtPlateForce = 4
rtPlateMoment = 5
rtPlateCurvature = 6
rtPlatePlyStress = 7
rtPlatePlyStrain = 8
rtPlatePlyReserve = 9
rtPlateFlux = 10
rtPlateGradient = 11
rtPlateRCDesign = 12
rtPlateCreepStrain = 13
rtPlateSoil = 14
rtPlateUser = 15
rtPlateNodeReact = 16
rtPlateNodeDisp = 17
rtPlateNodeBirthDisp = 18
rtPlateEffectiveStress = 19
rtPlateEffectiveForce = 20
rtPlateNodeFlux = 21
rtPlateTotalStrain = 22
rtPlateTotalCurvature = 23"""


    text4 = """aoRestraint = 1
aoForce = 2
aoMoment = 3
aoTemperature = 4
aoMTranslation = 5
aoMRotation = 6
aoKTranslation = 7
aoKRotation = 8
aoDamping = 9
aoNSMass = 10
aoNodeInfluence = 11
aoNodeHeatSource = 12
aoNodeVelocity = 13
aoNodeAcceleration = 14
aoVertexMeshSize = 20
aoBeamAngle = 21
aoBeamOffset = 22
aoBeamTEndRelease = 23
aoBeamREndRelease = 24
aoBeamSupport = 25
aoBeamPreTension = 26
aoCableFreeLength = 27
aoBeamDLL = 28
aoBeamDLG = 29
aoBeamCFL = 30
aoBeamCFG = 31
aoBeamCML = 32
aoBeamCMG = 33
aoBeamTempGradient = 34
aoBeamConvection = 35
aoBeamRadiation = 36
aoBeamFlux = 37
aoBeamHeatSource = 38
aoBeamRadius = 39
aoPipePressure = 40
aoBeamNSMass = 41
aoPipeTemperature = 42
aoBeamDML = 44
aoBeamStringGroup = 45
aoBeamPreCurvature = 46
aoBeamTaper = 92
aoBeamInfluence = 93
aoBeamSectionFactor = 94
aoBeamCreepLoadingAge = 95
aoBeamEndAttachment = 96
aoBeamConnectionUCS = 97
aoBeamStageProperty = 98
aoBeamSideAttachment = 120
aoPlateAngle = 51
aoPlateOffset = 52
aoPlatePreLoad = 53
aoPlateFacePressure = 54
aoPlateFaceShear = 55
aoPlateEdgeNormalPressure = 56
aoPlateEdgeShear = 57
aoPlateEdgeTransverseShear = 58
aoPlateTempGradient = 59
aoPlateEdgeSupport = 60
aoPlateFaceSupport = 61
aoPlateEdgeConvection = 62
aoPlateEdgeRadiation = 63
aoPlateFlux = 64
aoPlateHeatSource = 65
aoPlateGlobalPressure = 66
aoPlateEdgeRelease = 67
aoPlateReinforcement = 68
aoPlateThickness = 69
aoPlateNSMass = 70
aoLoadPatch = 71
aoPlateEdgeGlobalPressure = 72
aoPlatePreCurvature = 73
aoPlatePointForce = 99
aoPlatePointMoment = 100
aoPlateFaceConvection = 101
aoPlateFaceRadiation = 102
aoPlateInfluence = 103
aoPlateSoilStress = 104
aoPlateSoilRatio = 105
aoPlateCreepLoadingAge = 106
aoPlateEdgeAttachment = 107
aoPlateFaceAttachment = 108
aoPlateStageProperty = 109
aoPlateSectionFactor = 121
aoPlateCavity = 122
aoBrickPressure = 81
aoBrickShear = 82
aoBrickFaceFoundation = 83
aoBrickConvection = 84
aoBrickRadiation = 85
aoBrickFlux = 86
aoBrickHeatSource = 87
aoBrickGlobalPressure = 88
aoBrickNSMass = 89
aoBrickLocalAxes = 90
aoBrickPreLoad = 91
aoBrickPointForce = 110
aoBrickInfluence = 111
aoBrickSoilStress = 112
aoBrickSoilRatio = 113
aoBrickCreepLoadingAge = 114
aoBrickFaceAttachment = 115
aoBrickStageProperty = 116
aoBrickCavity = 123
aoPathPointForce = 117
aoPathDistributedForce = 118
aoPathHeatSource = 119"""

    make_enum(text4)

    tally()



if __name__ == "__main__":
    extract_final_images()
