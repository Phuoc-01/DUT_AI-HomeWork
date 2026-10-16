from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


class Member(ABC):
    def __init__(self, name : str, year_of_birth : int):
        self.name = name
        self.year_of_birth = year_of_birth
        
    @staticmethod
    def valid_birth_year(year_of_birth : int) -> bool:
        return 1950 <= year_of_birth <= 2015
    
    @property
    def name(self) -> str:
        return self._name
    @name.setter
    def name(self, value : str):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ten khong duoc rong")
        self._name = value.strip()
        
    @property
    def year_of_birth(self) -> int:
        return self._year_of_birth
    
    @year_of_birth.setter
    def year_of_birth(self, value : int):
        if not self.valid_birth_year(value):
            raise ValueError("Nam sinh phai trong khoang 1950 den 2015")
        self._year_of_birth = value

    @abstractmethod
    def role(self):
        pass
        
    @abstractmethod    
    def describe(self):
        pass
        
    @classmethod
    def from_dict(cls, data : dict):
        return cls(**data)
        
        
class StudentMember(Member):
    def __init__(self, name : str, year_of_birth : int, major : str):
        super().__init__(name, year_of_birth)
        self.major = major
        
    def role(self) -> str:
        return "Student"
    
    def describe(self) -> str:
        return f"Student Member: {self.name}, Year of Birth: {self.year_of_birth}, Major: {self.major}"


class Mentor(Member):
    def __init__(self, name : str, year_of_birth : int, expertise : str):
        super().__init__(name, year_of_birth)
        self.expertise = expertise

    def role(self) -> str:
        return "Mentor"

    def describe(self) -> str:
        return f"Mentor: {self.name}, Year of Birth: {self.year_of_birth}, Expertise: {self.expertise}"

class Engineer(Member):
    def __init__(self, name : str, year_of_birth : int, tech_stack : str):
        super().__init__(name, year_of_birth)
        self.tech_stack = tech_stack

    def role(self) -> str:
        return "Engineer"
    
    def describe(self) -> str:
        return f"Engineer: {self.name}, Year of Birth: {self.year_of_birth}, Tech Stack: {self.tech_stack}"
    
    
class LabGroup:
    def __init__(self, name : str):
        self.name = name
        self.members : List[Member] = []
        
    def add_member(self, member : Member):
        if not isinstance(member, Member):
            raise ValueError("Chi chap nhan doi tuong thuoc class Member")
        self.members.append(member)
            
    def sort_by_age (self):
        self.members.sort(key=lambda m: m.year_of_birth, reverse = True)
    
    def describe(self) -> str:
        des = [member.describe() for member in self.members]
        return f"LabGroup: {self.name}\n" + "\n".join(des)
    
    def average_birth_year_of_mentors(self) -> Optional[float]:
        mentors = [m for m in self.members if isinstance(m, Mentor)]
        if not mentors:
            return None
        return sum(m.year_of_birth for m in mentors) / len(mentors)
        
    def count_by_role(self, role : str) -> int:
        return sum(1 for member in self.members if member.role() == role)
    
    
    
class StudentInput(BaseModel):
    name : str = Field(min_length = 1)
    year_of_birth : int = Field(ge=1950, le=2015)
    expertise : str