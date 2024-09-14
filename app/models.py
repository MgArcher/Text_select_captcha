from pydantic import BaseModel
from typing import Optional, Union


class Input(BaseModel):
    dataType: int
    imageSource: str
    imageID: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "dataType": 1,
                "imageSource": "https://static.geetest.com/captcha_v3/batch/v3/33114/2023-04-23T10/word/2c4d5ce6e02e4cb883f45e6476be9ba7.jpg?challenge=b11d16e456aad5c622d244047cc389ea",
                "imageID": "string"
                }
        }


class OutPut(BaseModel):
    code: int
    msg: str
    data: Union[list, dict, str]

    class Config:
        schema_extra = {
            "example": {
                      "code": 200,
                      "msg": "成功",
                      "data": {
                        "imageID": "string",
                        "res": [
                          [
                            184,
                            0,
                            259,
                            67
                          ],
                          [
                            176,
                            238,
                            244,
                            310
                          ],
                          [
                            63,
                            70,
                            132,
                            142
                          ]
                        ]
                      }
                    }
        }

