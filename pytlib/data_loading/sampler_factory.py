# base class for input generation
# Source provides access to raw frames
# Source could be provide a sequence of frames in order (array of frames)
# Sould be able to query source for frames in order or any particular frame

# Perturber adds reconstructable perturbations to items coming of the source

# Sampler should take the source and perturber (optional?) turn into something the application wants

#### Data Loader factory for kind of dependency injection
from data_loading.sources.kitti_source import KITTISource
from data_loading.samplers.autoencoder_sampler import AutoEncoderSampler

class SamplerFactory:
    @staticmethod
    def GetAESampler(source,max_frames=200,obj_types=['Car'],crop_size=[100,100]):
        source = KITTISource(source,max_frames=max_frames)
        sampler_params = {'crop_size':crop_size,'obj_types':obj_types}
        return AutoEncoderSampler(source,sampler_params)
