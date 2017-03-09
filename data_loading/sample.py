# a sample should contain both the data and the optional targets
class Sample:
  def __init__(self,data,target=None):
    self.data = data
    self.target = target
