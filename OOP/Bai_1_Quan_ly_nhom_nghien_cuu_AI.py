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
    organization = 'DUT AI'
    
    def __init__(self, name : str):
        self.name = name
        self.members : List[Member] = []
        
    def add_member(self, member : Member):
        if not isinstance(member, Member):
            raise TypeError("Chi chap nhan doi tuong thuoc class Member")
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
    
    def to_report(self) -> 'LabReport':
        return LabReport(
            lab_name= self.name,
            total_members= len(self.members),
            student_count= self.count_by_role("Student"),
            mentor_count= self.count_by_role("Mentor"),
            engineer_count= self.count_by_role("Engineer")
        )
        
    @classmethod
    def from_members(cls, name : str, members : List[Member]) -> 'LabGroup':
        lab = cls(name)
        for member in members:
            lab.add_member(member)
        return lab
    
    @classmethod
    def from_dict(cls, data : dict):
        return cls(**data)
    
class StudentInput(BaseModel):
    name : str = Field(min_length = 1)
    year_of_birth : int = Field(ge=1950, le=2015)
    major : str
    
    @field_validator('name')
    def name_not_empty(cls, a):
        if not a.strip():
            raise ValueError('Ten khong rong')
        return a.strip()
    
    @classmethod
    def from_dict(cls, data : dict):
        return cls.model_validate(data)
    

class EngineerInput(BaseModel):
    name : str = Field(min_length = 1)
    year_of_birth : int = Field(ge=1950, le=2015)
    tech_stack : str
    
    @field_validator('name')
    def name_not_empty(cls, a):
        if not a.strip():
            raise ValueError('Ten khong rong')
        return a.strip()
    
    @classmethod
    def from_dict(cls, data : dict):
        return cls.model_validate(data)
    
class MentorInput(BaseModel):
    name : str = Field(min_length=1)
    year_of_birth : int = Field(ge=1950, le=2015)
    expertise : str
    
    @field_validator('name')
    def name_not_empty(cls, a):
        if not a.strip():
            raise ValueError('Ten khong rong')
        return a.strip()
    
    @classmethod
    def from_dict(cls, data : dict):
        return cls.model_validate(data)
    
@dataclass(frozen=True)
class LabReport:
    lab_name : str
    total_members : int
    student_count : int
    mentor_count : int
    engineer_count : int
    
    
if __name__ == "__main__":
    print("--- 1. Khởi tạo nhóm NLP Lab ---")
    members_list = [
        StudentMember('An', 2005, 'Data Science'),
        Mentor('Bình', 1990, 'NLP'),
        Mentor('Chi', 1988, 'Computer Vision'),
        Engineer('Dũng', 1999, 'Python')
    ]
    lab = LabGroup.from_members("NLP Lab", members_list)

    print(f"Tổ chức: {LabGroup.organization}")
    print(f"Đếm số lượng Mentor: {lab.count_by_role('Mentor')} (Kỳ vọng: 2)")
    
    avg_mentor_year = lab.average_birth_year_of_mentors()
    print(f"Trung bình năm sinh Mentor: {avg_mentor_year} (Kỳ vọng: 1989.0)")

    report = lab.to_report()
    print(f"Tổng số thành viên (từ report): {report.total_members} (Kỳ vọng: 4)")
    print(f"Chi tiết Report: {report}")

    print("\n--- 2. Kiểm tra sắp xếp theo tuổi ---")
    lab.sort_by_age()
    sorted_years = [m.year_of_birth for m in lab.members]
    print(f"Thứ tự năm sinh sau khi sắp xếp: {sorted_years} (Kỳ vọng: [2005, 1999, 1990, 1988])")

    print("\n--- 3. Kiểm tra tính đa hình (describe) ---")
    print(lab.describe())

    print("\n--- 4. Kiểm tra các trường hợp ngoại lệ (Exceptions) ---")
    
    try:
        StudentMember('Tèo', 2022, 'IT')
    except ValueError as e:
        print(f"Bắt lỗi ValueError thành công: {e}")

    try:
        StudentInput.from_dict({'name': 'Tèo', 'year_of_birth': 2022, 'major': 'IT'})
    except ValidationError as e:
        print(f"Bắt lỗi Pydantic Validation thành công.")

    try:
        lab.add_member('An')
    except TypeError as e:
        print(f"Bắt lỗi TypeError thành công: {e}")
        
    empty_lab = LabGroup("Empty Lab")
    print(f"\nTrung bình năm sinh Mentor của nhóm rỗng: {empty_lab.average_birth_year_of_mentors()} (Kỳ vọng: None)")