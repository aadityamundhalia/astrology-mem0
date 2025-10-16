from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text,
    Boolean, BigInteger, Date, Time, JSON
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)


Base.metadata.create_all(bind=engine)
