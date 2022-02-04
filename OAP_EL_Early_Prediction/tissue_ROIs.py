from owlready import *


# define a owl object
onto = Ontology("http://test.org/tissueROIs.owl")
onto_path.append("C:/Users/liz27/OneDrive/Documents/CCHMC/Early prediction/Cincinnati EPS/Data")


# main class
class Tissue(Thing):
    ontology = onto


# # Temporal lobe      
class Temporal_lobe(Tissue):
    pass

class rel_volume_AnteriortemporallobemedialpartleftGM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobemedialpartrightGM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobelateralpartleftGM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobelateralpartrightGM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobemedialpartleftWM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobemedialpartrightWM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobelateralpartleftWM(Temporal_lobe):
    pass
class rel_volume_AnteriortemporallobelateralpartrightWM(Temporal_lobe):
    pass
class thickness_Anteriortemporallobemedialpartleft(Temporal_lobe):
    pass
class thickness_Anteriortemporallobemedialpartright(Temporal_lobe):
    pass
class thickness_Anteriortemporallobelateralpartleft(Temporal_lobe):
    pass
class thickness_Anteriortemporallobelateralpartright(Temporal_lobe):
    pass
class sulc_Anteriortemporallobemedialpartleft(Temporal_lobe):
    pass
class sulc_Anteriortemporallobemedialpartright(Temporal_lobe):
    pass
class sulc_Anteriortemporallobelateralpartleft(Temporal_lobe):
    pass
class sulc_Anteriortemporallobelateralpartright(Temporal_lobe):
    pass
class curvature_Anteriortemporallobemedialpartleft(Temporal_lobe):
    pass
class curvature_Anteriortemporallobemedialpartright(Temporal_lobe):
    pass
class curvature_Anteriortemporallobelateralpartleft(Temporal_lobe):
    pass
class curvature_Anteriortemporallobelateralpartright(Temporal_lobe):
    pass
class GI_Anteriortemporallobemedialpartleft(Temporal_lobe):
    pass
class GI_Anteriortemporallobemedialpartright(Temporal_lobe):
    pass
class GI_Anteriortemporallobelateralpartleft(Temporal_lobe):
    pass
class GI_Anteriortemporallobelateralpartright(Temporal_lobe):
    pass
class rel_surface_area_Anteriortemporallobemedialpartleft(Temporal_lobe):
    pass
class rel_surface_area_Anteriortemporallobemedialpartright(Temporal_lobe):
    pass
class rel_surface_area_Anteriortemporallobelateralpartleft(Temporal_lobe):
    pass
class rel_surface_area_Anteriortemporallobelateralpartright(Temporal_lobe):
    pass
class rel_volume_TemporallobeleftGM_mergedregion_(Temporal_lobe):
    pass
class rel_volume_TemporalloberightGM_mergedregion_(Temporal_lobe):
    pass
class rel_volume_TemporallobeleftWM_mergedregion_(Temporal_lobe):
    pass
class rel_volume_TemporalloberightWM_mergedregion_(Temporal_lobe):
    pass
class thickness_Temporallobeleft_mergedregion_(Temporal_lobe):
    pass
class thickness_Temporalloberight_mergedregion_(Temporal_lobe):
    pass
class sulc_Temporallobeleft_mergedregion_(Temporal_lobe):
    pass
class sulc_Temporalloberight_mergedregion_(Temporal_lobe):
    pass
class curvature_Temporallobeleft_mergedregion_(Temporal_lobe):
    pass
class curvature_Temporalloberight_mergedregion_(Temporal_lobe):
    pass
class GI_Temporallobeleft_mergedregion_(Temporal_lobe):
    pass
class GI_Temporalloberight_mergedregion_(Temporal_lobe):
    pass
class rel_surface_area_Temporallobeleft_mergedregion_(Temporal_lobe):
    pass
class rel_surface_area_Temporalloberight_mergedregion_(Temporal_lobe):
    pass

# Posterior fossa  
class Posterior_fossa(Tissue):
    pass

class rel_volume_Cerebellum(Posterior_fossa):
    pass
class rel_volume_Cerebellumleft(Posterior_fossa):
    pass
class rel_volume_Cerebellumright(Posterior_fossa):
    pass
class rel_volume_Brainstem(Posterior_fossa):
    pass
class rel_volume_Brainstemspansthemidline(Posterior_fossa):
    pass

# Insula and cingulate gyri
class Insula_cingulate_gyri(Tissue):
    pass

class rel_volume_InsularightGM(Insula_cingulate_gyri):
    pass
class rel_volume_InsulaleftGM(Insula_cingulate_gyri):
    pass
class rel_volume_InsularightWM(Insula_cingulate_gyri):
    pass
class rel_volume_InsulaleftWM(Insula_cingulate_gyri):
    pass
class thickness_Insularight(Insula_cingulate_gyri):
    pass
class thickness_Insulaleft(Insula_cingulate_gyri):
    pass
class sulc_Insularight(Insula_cingulate_gyri):
    pass
class sulc_Insulaleft(Insula_cingulate_gyri):
    pass
class curvature_Insularight(Insula_cingulate_gyri):
    pass
class curvature_Insulaleft(Insula_cingulate_gyri):
    pass
class rel_surface_area_Insularight(Insula_cingulate_gyri):
    pass
class rel_surface_area_Insulaleft(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusanteriorpartrightGM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusanteriorpartleftGM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusposteriorpartrightGM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusposteriorpartleftGM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusanteriorpartrightWM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusanteriorpartleftWM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusposteriorpartrightWM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusposteriorpartleftWM(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusleftGM_mergedregion_(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusrightGM_mergedregion_(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusleftWM_mergedregion_(Insula_cingulate_gyri):
    pass
class rel_volume_CingulategyrusrightWM_mergedregion_(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusanteriorpartright(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusanteriorpartleft(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusposteriorpartright(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusposteriorpartleft(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusleft_mergedregion_(Insula_cingulate_gyri):
    pass
class thickness_Cingulategyrusright_mergedregion_(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusanteriorpartright(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusanteriorpartleft(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusposteriorpartright(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusposteriorpartleft(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusleft_mergedregion_(Insula_cingulate_gyri):
    pass
class sulc_Cingulategyrusright_mergedregion_(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusanteriorpartright(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusanteriorpartleft(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusposteriorpartright(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusposteriorpartleft(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusleft_mergedregion_(Insula_cingulate_gyri):
    pass
class curvature_Cingulategyrusright_mergedregion_(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusanteriorpartright(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusanteriorpartleft(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusposteriorpartright(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusposteriorpartleft(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusleft_mergedregion_(Insula_cingulate_gyri):
    pass
class GI_Cingulategyrusright_mergedregion_(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusanteriorpartright(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusanteriorpartleft(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusposteriorpartright(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusposteriorpartleft(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusleft_mergedregion_(Insula_cingulate_gyri):
    pass
class rel_surface_area_Cingulategyrusright_mergedregion_(Insula_cingulate_gyri):
    pass

# Frontal lobe class
class Frontallobe(Tissue):
    pass

class rel_volume_FrontalloberightGM(Frontallobe):
    pass
class rel_volume_FrontallobeleftGM(Frontallobe):
    pass
class rel_volume_FrontalloberightWM(Frontallobe):
    pass
class thickness_Frontalloberight(Frontallobe):
    pass
class thickness_Frontallobeleft(Frontallobe):
    pass
class sulc_Frontalloberight(Frontallobe):
    pass
class sulc_Frontallobeleft(Frontallobe):
    pass
class curvature_Frontalloberight(Frontallobe):
    pass
class curvature_Frontallobeleft(Frontallobe):
    pass
class GI_Frontalloberight(Frontallobe):
    pass
class GI_Frontallobeleft(Frontallobe):
    pass
class rel_surface_area_Frontalloberight(Frontallobe):
    pass
class rel_surface_area_Frontallobeleft(Frontallobe):
    pass


# Occipitallobe
class Occipitallobe(Tissue):
    pass

class rel_volume_OccipitalloberightGM(Occipitallobe):
    pass
class rel_volume_OccipitallobeleftGM(Occipitallobe):
    pass
class rel_volume_OccipitalloberightWM(Occipitallobe):
    pass
class rel_volume_OccipitallobeleftWM(Occipitallobe):
    pass
class thickness_Occipitalloberight(Occipitallobe):
    pass
class thickness_Occipitallobeleft(Occipitallobe):
    pass
class sulc_Occipitalloberight(Occipitallobe):
    pass
class sulc_Occipitallobeleft(Occipitallobe):
    pass
class curvature_Occipitalloberight(Occipitallobe):
    pass
class curvature_Occipitallobeleft(Occipitallobe):
    pass
class GI_Occipitalloberight(Occipitallobe):
    pass
class GI_Occipitallobeleft(Occipitallobe):
    pass
class rel_surface_area_Occipitalloberight(Occipitallobe):
    pass
class rel_surface_area_Occipitallobeleft(Occipitallobe):
    pass


# Parietallobe
class Parietallobe(Tissue):
    pass

class rel_volume_ParietalloberightGM(Parietallobe):
    pass
class rel_volume_ParietallobeleftGM(Parietallobe):
    pass
class rel_volume_ParietalloberightWM(Parietallobe):
    pass
class rel_volume_ParietallobeleftWM(Parietallobe):
    pass
class thickness_Parietalloberight(Parietallobe):
    pass
class thickness_Parietallobeleft(Parietallobe):
    pass
class sulc_Parietalloberight(Parietallobe):
    pass
class sulc_Parietallobeleft(Parietallobe):
    pass
class curvature_Parietalloberight(Parietallobe):
    pass
class curvature_Parietallobeleft(Parietallobe):
    pass
class GI_Parietalloberight(Parietallobe):
    pass
class GI_Parietallobeleft(Parietallobe):
    pass
class rel_surface_area_Parietalloberight(Parietallobe):
    pass
class rel_surface_area_Parietallobeleft(Parietallobe):
    pass

# Basal ganglia
class Basal_ganglia(Tissue):
    pass

class rel_volume_Caudatenucleusright(Basal_ganglia):
    pass
class rel_volume_Caudatenucleusleft(Basal_ganglia):
    pass
class rel_volume_ThalamusrighthighintensitypartinT2(Basal_ganglia):
    pass
class rel_volume_ThalamuslefthighintensitypartinT2(Basal_ganglia):
    pass
class rel_volume_ThalamusrightlowintensitypartinT2(Basal_ganglia):
    pass
class rel_volume_ThalamusleftlowintensitypartinT2(Basal_ganglia):
    pass
class rel_volume_Subthalamicnucleusright(Basal_ganglia):
    pass
class rel_volume_Subthalamicnucleusleft(Basal_ganglia):
    pass
class rel_volume_LentiformNucleusright(Basal_ganglia):
    pass
class rel_volume_LentiformNucleusleft(Basal_ganglia):
    pass


# Corpus callosum
class Corpus_callosum(Tissue):
    pass
class rel_volume_CorpusCallosum(Corpus_callosum):
    pass

# Ventricles
class Ventricles(Tissue):
    pass

class rel_volume_LateralVentricleleft(Ventricles):
    pass
class rel_volume_LateralVentricleright(Ventricles):
    pass


onto.save()

