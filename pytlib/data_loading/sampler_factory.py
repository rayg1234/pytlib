#### Data Loader factory for kind of dependency injection
# this is not really neccessary now, user should probably just create these directly in the config file

from data_loading.sources.kitti_source import KITTISource
from data_loading.samplers.autoencoder_sampler import AutoEncoderSampler

class SamplerFactory:
    @staticmethod
    def GetAESampler(source,max_frames=200,obj_types=['Car'],crop_size=[100,100]):
        source = KITTISource(source,max_frames=max_frames)
        sampler_params = {'crop_size':crop_size,'obj_types':obj_types}
        return AutoEncoderSampler(source,sampler_params)
