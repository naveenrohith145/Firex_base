import base64
from db import db
from models.feedback import Feedback

class FeedbackDao:

    @staticmethod
    def get_feedback():
        connection = db.session.connection().connection
        cursor=connection.cursor()
        data=Feedback.query.filter(Feedback.feedback.isnot(None)).all()
        all_data=[]
        for i in data:
            all_data.append({
                "id": i.id,
                "user_prompt": i.user_prompt,
                "response": i.response,
                "feedback": i.feedback,
                "user_id": 2,
            })
        cursor.close()
        connection.close()

        return all_data

    @staticmethod
    def feedback_by_id(record_id):
        return Feedback.query.filter(Feedback.id==record_id, Feedback.feedback.isnot(None)).first()

    @staticmethod
    def update_feedback(record_id, fb_value):
        feedback_record = db.session.query(Feedback).filter_by(id=record_id).first()
        if feedback_record:
            feedback_record.feedback = fb_value
            db.session.commit()
            return feedback_record
        return None

    @staticmethod
    def insert(user_prompt, response,feedback,user_id):
        feedback_entry = Feedback(
            user_prompt=user_prompt,
            response=response,
            feedback=feedback,
            user_id=user_id,

        )
        db.session.add(feedback_entry)
        db.session.commit()

        return feedback_entry

