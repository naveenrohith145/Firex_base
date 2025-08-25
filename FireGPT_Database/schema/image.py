from marshmallow import Schema, fields

class ImageSchema(Schema):
    id=fields.Int(dump_only=True)
    image_description=fields.Str(dump_only=True)
    images = fields.Int(dump_only=True)
    user_id=fields.Int(dump_only=True)

class ImageBinarySchema(Schema):
    id = fields.Int(dump_only=True)
    image_description = fields.Str(dump_only=True)
    user_id=fields.Int(dump_only=True)
    # image = fields.Int(dump_only=True)
    # image_large_binary = fields.Raw(dump_only=True)
