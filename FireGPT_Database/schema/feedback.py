from marshmallow import Schema, fields

class FeedbackSchema(Schema):
    id=fields.Int(dump_only=True)
    user_id=fields.Int(required=True)
    user_prompt=fields.Str(required=True)
    response=fields.Str(required=True)
    feedback=fields.Bool(required=True)
    images = fields.Int(required=True)