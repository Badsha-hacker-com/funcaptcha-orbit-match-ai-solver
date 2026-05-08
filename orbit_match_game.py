import os

import funcaptcha_challenger.model
from PIL import Image
from funcaptcha_challenger.model import BaseModel
from funcaptcha_challenger.predictor import ImagePairClassifierPredictor


class PredictOrbitMatchGame(ImagePairClassifierPredictor):


    def _get_model(self):
        return BaseModel("orbit_match_game.onnx")

    def is_support(self, variant, instruction):
        return 'orbit_match_game' == variant


if __name__ == '__main__':
    funcaptcha_challenger.model.auto_update = False
    funcaptcha_challenger.model.model_root_path = os.path.dirname(os.path.abspath(__file__))
    funcaptcha_challenger.predictors.append(PredictOrbitMatchGame())

    image_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'image')

    for image in os.listdir(image_dir):
        if not image.endswith('.jpg'):
            continue
        image_path = os.path.join(image_dir, image)

        image = Image.open(image_path)
        predict_index = funcaptcha_challenger.predict(image,'orbit_match_game')


        print(f"{image_path} ans is : {predict_index}" )
        # ans_path = image_path.replace('.jpg', '.txt')
        # with open(ans_path, 'r') as f:
        #     ans = f.read()

        # print(f'predict: {predict_index == int(ans)}')
