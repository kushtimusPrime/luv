from abc import ABCMeta, abstractmethod


class Camera(metaclass=ABCMeta):

    @abstractmethod
	def capture_image(self):
		pass
