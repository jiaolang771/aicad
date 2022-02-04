import os
from owlready import *


# define a owl object
onto = Ontology("http://test.org/CorticalMaturation.owl")
onto_path.append("C:/Users/liz27/OneDrive/Documents/CCHMC/Early prediction/Cincinnati EPS/Data")

# main class
class CorticalMaturation(Thing):
    ontology = onto

class Volume(CorticalMaturation):
    pass
class Thickness(CorticalMaturation):
    pass
class Sulcal_depth(CorticalMaturation):
    pass
class Curvature(CorticalMaturation):
    pass
class Gyrification_index(CorticalMaturation):
    pass
class Surface_area(CorticalMaturation):
    pass

# main class
class Geometric(Thing):
    ontology = onto

# subclass
class rel_volume_CSF(Geometric):
    pass

class rel_volume_Corticalgraymatter(Geometric):
    pass

class rel_volume_Whitematter(Geometric):
    pass

class rel_volume_Background(Geometric):
    pass

class rel_volume_Ventricles(Geometric):
    pass

class rel_volume_Cerebellum(Geometric):
    pass

class rel_volume_DeepGrayMatter(Geometric):
    pass

class rel_volume_Brainstem(Geometric):
    pass

class rel_volume_HippocampiandAmygdala(Geometric):
    pass

class rel_volume_Hippocampusleft(Geometric):
    pass

class rel_volume_Hippocampusright(Geometric):
    pass

class rel_volume_Amygdalaleft(Geometric):
    pass

class rel_volume_Amygdalaright(Geometric):
    pass

class rel_volume_AnteriortemporallobemedialpartleftGM(Geometric):
    pass

class rel_volume_AnteriortemporallobemedialpartrightGM(Geometric):
    pass

class rel_volume_AnteriortemporallobelateralpartleftGM(Geometric):
    pass

class rel_volume_AnteriortemporallobelateralpartrightGM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensanteriorpartleftGM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensanteriorpartrightGM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusmiddlepartleftGM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusmiddlepartrightGM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrianteriorpartleftGM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrianteriorpartrightGM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleftGM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisanteriorpartrightGM(Geometric):
    pass

class rel_volume_Cerebellumleft(Geometric):
    pass

class rel_volume_Cerebellumright(Geometric):
    pass

class rel_volume_Brainstemspansthemidline(Geometric):
    pass

class rel_volume_InsularightGM(Geometric):
    pass


class rel_volume_InsulaleftGM(Geometric):
    pass

class rel_volume_OccipitalloberightGM(Geometric):
    pass

class rel_volume_OccipitallobeleftGM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensposteriorpartrightGM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensposteriorpartleftGM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisposteriorpartrightGM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleftGM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyriposteriorpartrightGM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyriposteriorpartleftGM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusposteriorpartrightGM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusposteriorpartleftGM(Geometric):
    pass

class rel_volume_CingulategyrusanteriorpartrightGM(Geometric):
    pass
class rel_volume_CingulategyrusanteriorpartleftGM(Geometric):
    pass

class rel_volume_CingulategyrusposteriorpartrightGM(Geometric):
    pass

class rel_volume_CingulategyrusposteriorpartleftGM(Geometric):
    pass
class rel_volume_FrontalloberightGM(Geometric):
    pass

class rel_volume_FrontallobeleftGM(Geometric):
    pass

class rel_volume_ParietalloberightGM(Geometric):
    pass
class rel_volume_ParietallobeleftGM(Geometric):
    pass

class rel_volume_Caudatenucleusright(Geometric):
    pass

class rel_volume_Caudatenucleusleft(Geometric):
    pass
class rel_volume_ThalamusrighthighintensitypartinT2(Geometric):
    pass

class rel_volume_ThalamuslefthighintensitypartinT2(Geometric):
    pass

class rel_volume_Subthalamicnucleusright(Geometric):
    pass
class rel_volume_Subthalamicnucleusleft(Geometric):
    pass

class rel_volume_LentiformNucleusright(Geometric):
    pass

class rel_volume_LentiformNucleusleft(Geometric):
    pass
class rel_volume_CorpusCallosum(Geometric):
    pass

class rel_volume_LateralVentricleleft(Geometric):
    pass

class rel_volume_LateralVentricleright(Geometric):
    pass

class rel_volume_AnteriortemporallobemedialpartleftWM(Geometric):
    pass

class rel_volume_AnteriortemporallobemedialpartrightWM(Geometric):
    pass

class rel_volume_AnteriortemporallobelateralpartleftWM(Geometric):
    pass

class rel_volume_AnteriortemporallobelateralpartrightWM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensanteriorpartleftWM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensanteriorpartrightWM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusmiddlepartleftWM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusmiddlepartrightWM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrianteriorpartleftWM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrianteriorpartrightWM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleftWM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisanteriorpartrightWM(Geometric):
    pass

class rel_volume_InsularightWM(Geometric):
    pass

class rel_volume_InsulaleftWM(Geometric):
    pass

class rel_volume_OccipitalloberightWM(Geometric):
    pass

class rel_volume_OccipitallobeleftWM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensposteriorpartrightWM(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensposteriorpartleftWM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisposteriorpartrightWM(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleftWM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyriposteriorpartrightWM(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyriposteriorpartleftWM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusposteriorpartrightWM(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusposteriorpartleftWM(Geometric):
    pass

class rel_volume_CingulategyrusanteriorpartrightWM(Geometric):
    pass

class rel_volume_CingulategyrusanteriorpartleftWM(Geometric):
    pass

class rel_volume_CingulategyrusposteriorpartrightWM(Geometric):
    pass

class rel_volume_CingulategyrusposteriorpartleftWM(Geometric):
    pass

class rel_volume_FrontalloberightWM(Geometric):
    pass

class rel_volume_FrontallobeleftWM(Geometric):
    pass

class rel_volume_ParietalloberightWM(Geometric):
    pass

class rel_volume_ParietallobeleftWM(Geometric):
    pass

class rel_volume_CSF_1(Geometric):
    pass

class rel_volume_Extra_cranialbackground(Geometric):
    pass

class rel_volume_Intra_cranialbackground(Geometric):
    pass

class rel_volume_ThalamusrightlowintensitypartinT2(Geometric):
    pass

class rel_volume_ThalamusleftlowintensitypartinT2(Geometric):
    pass

class rel_volume_TemporallobeleftGM_mergedregion_(Geometric):
    pass

class rel_volume_TemporalloberightGM_mergedregion_(Geometric):
    pass

class rel_volume_TemporallobeleftWM_mergedregion_(Geometric):
    pass

class rel_volume_TemporalloberightWM_mergedregion_(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusleftGM_mergedregion_(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusrightGM_mergedregion_(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusleftWM_mergedregion_(Geometric):
    pass

class rel_volume_SuperiortemporalgyrusrightWM_mergedregion_(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrileftGM_mergedregion_(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrirightGM_mergedregion_(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrileftWM_mergedregion_(Geometric):
    pass

class rel_volume_MedialandinferiortemporalgyrirightWM_mergedregion_(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisleftGM_mergedregion_(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisrightGM_mergedregion_(Geometric):
    pass
class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisleftWM_mergedregion_(Geometric):
    pass

class rel_volume_LateraloccipitotemporalgyrusgyrusfusiformisrightWM_mergedregion_(Geometric):
    pass

class rel_volume_CingulategyrusleftGM_mergedregion_(Geometric):
    pass

class rel_volume_CingulategyrusrightGM_mergedregion_(Geometric):
    pass

class rel_volume_CingulategyrusleftWM_mergedregion_(Geometric):
    pass

class rel_volume_CingulategyrusrightWM_mergedregion_(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensleftGM_mergedregion_(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensrightGM_mergedregion_(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensleftWM_mergedregion_(Geometric):
    pass

class rel_volume_GyriparahippocampalisetambiensrightWM_mergedregion_(Geometric):
    pass

class thickness_Anteriortemporallobemedialpartleft(Geometric):
    pass

class thickness_Anteriortemporallobemedialpartright(Geometric):
    pass

class thickness_Anteriortemporallobelateralpartright(Geometric):
    pass

class thickness_Gyriparahippocampalisetambiensanteriorpartleft(Geometric):
    pass

class thickness_Gyriparahippocampalisetambiensanteriorpartright(Geometric):
    pass

class thickness_Superiortemporalgyrusmiddlepartleft(Geometric):
    pass

class thickness_Superiortemporalgyrusmiddlepartright(Geometric):
    pass

class thickness_Medialandinferiortemporalgyrianteriorpartleft(Geometric):
    pass

class thickness_Medialandinferiortemporalgyrianteriorpartright(Geometric):
    pass

class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleft(Geometric):
    pass

class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartright(Geometric):
    pass

class thickness_Insularight(Geometric):
    pass

class thickness_Insulaleft(Geometric):
    pass

class thickness_Occipitalloberight(Geometric):
    pass

class thickness_Occipitallobeleft(Geometric):
    pass

class thickness_Gyriparahippocampalisetambiensposteriorpartright(Geometric):
    pass

class thickness_Gyriparahippocampalisetambiensposteriorpartleft(Geometric):
    pass

class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartright(Geometric):
    pass

class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleft(Geometric):
    pass

class thickness_Medialandinferiortemporalgyriposteriorpartright(Geometric):
    pass

class thickness_Medialandinferiortemporalgyriposteriorpartleft(Geometric):
    pass

class thickness_Superiortemporalgyrusposteriorpartright(Geometric):
    pass

class thickness_Superiortemporalgyrusposteriorpartleft(Geometric):
    pass

class thickness_Cingulategyrusanteriorpartright(Geometric):
    pass

class thickness_Cingulategyrusanteriorpartleft(Geometric):
    pass

class thickness_Cingulategyrusposteriorpartright(Geometric):
    pass

class thickness_Cingulategyrusposteriorpartleft(Geometric):
    pass

class thickness_Frontalloberight(Geometric):
    pass

class thickness_Frontallobeleft(Geometric):
    pass

class thickness_Parietalloberight(Geometric):
    pass

class thickness_Parietallobeleft(Geometric):
    pass
class thickness_Temporallobeleft_mergedregion_(Geometric):
    pass
class thickness_Temporalloberight_mergedregion_(Geometric):
    pass
class thickness_Superiortemporalgyrusleft_mergedregion_(Geometric):
    pass
class thickness_Superiortemporalgyrusright_mergedregion_(Geometric):
    pass
class thickness_Medialandinferiortemporalgyrileft_mergedregion_(Geometric):
    pass
class thickness_Medialandinferiortemporalgyriright_mergedregion_(Geometric):
    pass
class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisleft_mergedregion_(Geometric):
    pass
class thickness_Lateraloccipitotemporalgyrusgyrusfusiformisright_mergedregion_(Geometric):
    pass
class thickness_Cingulategyrusleft_mergedregion_(Geometric):
    pass
class thickness_Cingulategyrusright_mergedregion_(Geometric):
    pass
class thickness_Gyriparahippocampalisetambiensleft_mergedregion_(Geometric):
    pass
class thickness_Gyriparahippocampalisetambiensright_mergedregion_(Geometric):
    pass
class sulc_Anteriortemporallobemedialpartleft(Geometric):
    pass
class sulc_Anteriortemporallobemedialpartright(Geometric):
    pass
class sulc_Anteriortemporallobelateralpartleft(Geometric):
    pass
class sulc_Anteriortemporallobelateralpartright(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensanteriorpartleft(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensanteriorpartright(Geometric):
    pass
class sulc_Superiortemporalgyrusmiddlepartleft(Geometric):
    pass
class sulc_Superiortemporalgyrusmiddlepartright(Geometric):
    pass
class sulc_Medialandinferiortemporalgyrianteriorpartleft(Geometric):
    pass
class sulc_Medialandinferiortemporalgyrianteriorpartright(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleft(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartright(Geometric):
    pass
class sulc_Insularight(Geometric):
    pass
class sulc_Insulaleft(Geometric):
    pass
class sulc_Occipitalloberight(Geometric):
    pass
class sulc_Occipitallobeleft(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensposteriorpartright(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensposteriorpartleft(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartright(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleft(Geometric):
    pass
class sulc_Medialandinferiortemporalgyriposteriorpartright(Geometric):
    pass
class sulc_Medialandinferiortemporalgyriposteriorpartleft(Geometric):
    pass
class sulc_Superiortemporalgyrusposteriorpartright(Geometric):
    pass
class sulc_Superiortemporalgyrusposteriorpartleft(Geometric):
    pass
class sulc_Cingulategyrusanteriorpartright(Geometric):
    pass




class sulc_Cingulategyrusanteriorpartleft(Geometric):
    pass
class sulc_Cingulategyrusposteriorpartright(Geometric):
    pass
class sulc_Cingulategyrusposteriorpartleft(Geometric):
    pass
class sulc_Frontalloberight(Geometric):
    pass
class sulc_Frontallobeleft(Geometric):
    pass
class sulc_Parietalloberight(Geometric):
    pass
class sulc_Parietallobeleft(Geometric):
    pass
class sulc_Temporallobeleft_mergedregion_(Geometric):
    pass

class sulc_Temporalloberight_mergedregion_(Geometric):
    pass
class sulc_Superiortemporalgyrusleft_mergedregion_(Geometric):
    pass
class sulc_Superiortemporalgyrusright_mergedregion_(Geometric):
    pass
class sulc_Medialandinferiortemporalgyrileft_mergedregion_(Geometric):
    pass
class sulc_Medialandinferiortemporalgyriright_mergedregion_(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisleft_mergedregion_(Geometric):
    pass
class sulc_Lateraloccipitotemporalgyrusgyrusfusiformisright_mergedregion_(Geometric):
    pass
class sulc_Cingulategyrusleft_mergedregion_(Geometric):
    pass

class sulc_Cingulategyrusright_mergedregion_(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensleft_mergedregion_(Geometric):
    pass
class sulc_Gyriparahippocampalisetambiensright_mergedregion_(Geometric):
    pass
class curvature_Anteriortemporallobemedialpartleft(Geometric):
    pass
class curvature_Anteriortemporallobemedialpartright(Geometric):
    pass
class curvature_Anteriortemporallobelateralpartleft(Geometric):
    pass
class curvature_Anteriortemporallobelateralpartright(Geometric):
    pass
class curvature_Gyriparahippocampalisetambiensanteriorpartleft(Geometric):
    pass

class curvature_Gyriparahippocampalisetambiensanteriorpartright(Geometric):
    pass
class curvature_Superiortemporalgyrusmiddlepartleft(Geometric):
    pass
class curvature_Superiortemporalgyrusmiddlepartright(Geometric):
    pass
class curvature_Medialandinferiortemporalgyrianteriorpartleft(Geometric):
    pass
class curvature_Medialandinferiortemporalgyrianteriorpartright(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleft(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartright(Geometric):
    pass
class curvature_Insularight(Geometric):
    pass

class curvature_Insulaleft(Geometric):
    pass
class curvature_Occipitalloberight(Geometric):
    pass
class curvature_Occipitallobeleft(Geometric):
    pass
class curvature_Gyriparahippocampalisetambiensposteriorpartright(Geometric):
    pass
class curvature_Gyriparahippocampalisetambiensposteriorpartleft(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartright(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleft(Geometric):
    pass
class curvature_Medialandinferiortemporalgyriposteriorpartright(Geometric):
    pass

class curvature_Medialandinferiortemporalgyriposteriorpartleft(Geometric):
    pass
class curvature_Superiortemporalgyrusposteriorpartright(Geometric):
    pass
class curvature_Superiortemporalgyrusposteriorpartleft(Geometric):
    pass
class curvature_Cingulategyrusanteriorpartright(Geometric):
    pass
class curvature_Cingulategyrusanteriorpartleft(Geometric):
    pass
class curvature_Cingulategyrusposteriorpartright(Geometric):
    pass
class curvature_Cingulategyrusposteriorpartleft(Geometric):
    pass
class curvature_Frontalloberight(Geometric):
    pass

class curvature_Frontallobeleft(Geometric):
    pass
class curvature_Parietalloberight(Geometric):
    pass
class curvature_Parietallobeleft(Geometric):
    pass
class curvature_Temporallobeleft_mergedregion_(Geometric):
    pass
class curvature_Temporalloberight_mergedregion_(Geometric):
    pass
class curvature_Superiortemporalgyrusleft_mergedregion_(Geometric):
    pass
class curvature_Superiortemporalgyrusright_mergedregion_(Geometric):
    pass
class curvature_Medialandinferiortemporalgyrileft_mergedregion_(Geometric):
    pass
class curvature_Medialandinferiortemporalgyriright_mergedregion_(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisleft_mergedregion_(Geometric):
    pass
class curvature_Lateraloccipitotemporalgyrusgyrusfusiformisright_mergedregion_(Geometric):
    pass
class curvature_Cingulategyrusleft_mergedregion_(Geometric):
    pass
class curvature_Cingulategyrusright_mergedregion_(Geometric):
    pass
class curvature_Gyriparahippocampalisetambiensleft_mergedregion_(Geometric):
    pass
class curvature_Gyriparahippocampalisetambiensright_mergedregion_(Geometric):
    pass
class GI_Anteriortemporallobemedialpartleft(Geometric):
    pass
class GI_Anteriortemporallobemedialpartright(Geometric):
    pass
class GI_Anteriortemporallobelateralpartleft(Geometric):
    pass
class GI_Anteriortemporallobelateralpartright(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensanteriorpartleft(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensanteriorpartright(Geometric):
    pass
class GI_Superiortemporalgyrusmiddlepartleft(Geometric):
    pass
class GI_Superiortemporalgyrusmiddlepartright(Geometric):
    pass
class GI_Medialandinferiortemporalgyrianteriorpartleft(Geometric):
    pass
class GI_Medialandinferiortemporalgyrianteriorpartright(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleft(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartright(Geometric):
    pass
class GI_Occipitalloberight(Geometric):
    pass
class GI_Occipitallobeleft(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensposteriorpartright(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensposteriorpartleft(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartright(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleft(Geometric):
    pass
class GI_Medialandinferiortemporalgyriposteriorpartright(Geometric):
    pass
class GI_Medialandinferiortemporalgyriposteriorpartleft(Geometric):
    pass
class GI_Superiortemporalgyrusposteriorpartright(Geometric):
    pass
class GI_Superiortemporalgyrusposteriorpartleft(Geometric):
    pass
class GI_Cingulategyrusanteriorpartright(Geometric):
    pass
class GI_Cingulategyrusanteriorpartleft(Geometric):
    pass
class GI_Cingulategyrusposteriorpartright(Geometric):
    pass
class GI_Cingulategyrusposteriorpartleft(Geometric):
    pass
class GI_Frontalloberight(Geometric):
    pass
class GI_Frontallobeleft(Geometric):
    pass
class GI_Parietalloberight(Geometric):
    pass
class GI_Parietallobeleft(Geometric):
    pass
class GI_Temporallobeleft_mergedregion_(Geometric):
    pass
class GI_Temporalloberight_mergedregion_(Geometric):
    pass
class GI_Superiortemporalgyrusleft_mergedregion_(Geometric):
    pass
class GI_Superiortemporalgyrusright_mergedregion_(Geometric):
    pass
class GI_Medialandinferiortemporalgyrileft_mergedregion_(Geometric):
    pass
class GI_Medialandinferiortemporalgyriright_mergedregion_(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisleft_mergedregion_(Geometric):
    pass
class GI_Lateraloccipitotemporalgyrusgyrusfusiformisright_mergedregion_(Geometric):
    pass
class GI_Cingulategyrusleft_mergedregion_(Geometric):
    pass
class GI_Cingulategyrusright_mergedregion_(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensleft_mergedregion_(Geometric):
    pass
class GI_Gyriparahippocampalisetambiensright_mergedregion_(Geometric):
    pass
class rel_surface_area_Anteriortemporallobemedialpartleft(Geometric):
    pass
class rel_surface_area_Anteriortemporallobemedialpartright(Geometric):
    pass
class rel_surface_area_Anteriortemporallobelateralpartleft(Geometric):
    pass
class rel_surface_area_Anteriortemporallobelateralpartright(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensanteriorpartleft(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensanteriorpartright(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusmiddlepartleft(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusmiddlepartright(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyrianteriorpartleft(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyrianteriorpartright(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemporalgyrusgyrusfusiformisanteriorpartleft(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemtemporalporalgyrusgyrusfusiformisanteriorpartright(Geometric):
    pass
class rel_surface_area_Insularight(Geometric):
    pass
class rel_surface_area_Insulaleft(Geometric):
    pass
class rel_surface_area_Occipitalloberight(Geometric):
    pass
class rel_surface_area_Occipitallobeleft(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensposteriorpartright(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensposteriorpartleft(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartright(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemporalgyrusgyrusfusiformisposteriorpartleft(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyriposteriorpartright(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyriposteriorpartleft(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusposteriorpartright(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusposteriorpartleft(Geometric):
    pass
class rel_surface_area_Cingulategyrusanteriorpartright(Geometric):
    pass
class rel_surface_area_Cingulategyrusanteriorpartleft(Geometric):
    pass
class rel_surface_area_Cingulategyrusposteriorpartright(Geometric):
    pass
class rel_surface_area_Cingulategyrusposteriorpartleft(Geometric):
    pass
class rel_surface_area_Frontalloberight(Geometric):
    pass
class rel_surface_area_Frontallobeleft(Geometric):
    pass
class rel_surface_area_Parietalloberight(Geometric):
    pass
class rel_surface_area_Parietallobeleft(Geometric):
    pass
class rel_surface_area_Temporallobeleft_mergedregion_(Geometric):
    pass
class rel_surface_area_Temporalloberight_mergedregion_(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusleft_mergedregion_(Geometric):
    pass
class rel_surface_area_Superiortemporalgyrusright_mergedregion_(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyrileft_mergedregion_(Geometric):
    pass
class rel_surface_area_Medialandinferiortemporalgyriright_mergedregion_(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemporalgyrusgyrusfusiformisleft_mergedregion_(Geometric):
    pass
class rel_surface_area_Lateraloccipitotemporalgyrusgyrusfusiformisright_mergedregion_(Geometric):
    pass
class rel_surface_area_Cingulategyrusleft_mergedregion_(Geometric):
    pass
class rel_surface_area_Cingulategyrusright_mergedregion_(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensleft_mergedregion_(Geometric):
    pass
class rel_surface_area_Gyriparahippocampalisetambiensright_mergedregion_(Geometric):
    pass

onto.save()