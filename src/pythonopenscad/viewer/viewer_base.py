

from abc import ABC, abstractmethod

from pyglm import glm


class ViewerBase(ABC):
    
    @abstractmethod
    def get_projection_mat(self) -> glm.mat4:
        pass

    @abstractmethod
    def get_view_mat(self) -> glm.mat4:
        pass

    @abstractmethod
    def get_model_mat(self) -> glm.mat4:
        pass

    def get_mvp_mat(self) -> glm.mat4:
        return self.get_projection_mat() * self.get_view_mat() * self.get_model_mat()

    @abstractmethod
    def get_current_window_dims(self) -> tuple[int, int]:
        pass

    
