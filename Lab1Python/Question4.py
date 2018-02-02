'''
class Student:
    def __init__(self, first, last):
        self.first = first
        self.last = last
        self.classes = []

    def fullName(self):
        return '{} {}'.format(self.first,self.last)

class Classes(Student):
    def setName(self, className):

    def __init__(self, students = None):
        if students is None:
            self.students=[]
        else:
            self.students = students

    def add_Student(self, student):
        if student not in self.students:
            self.students.append(student)

    def remove_Student(self, student):
        if student not in self.students:
            self.student.remove(student)

    def print_Students(self, students):
        for student in self.students:
            print('->', student.students())

class Web_Application(Student):
    def __init__(self, first, last, students = None):
        super().__init__(first, last)


Python = Classes()
student1 = Python('Mr', 'Pickles', [1])

add_Student(student1)
'''

def sepLists(list1, list2):
    bothClasses = []
    eitherOr = []
    for i in range(0,len(list1)):
        for j in range(0,len(list2)):
            if list1[i] == list2[j]:
                bothClasses.append(list1[i])
            else:
                if list1[i] not in eitherOr:
                    eitherOr.append(list1[i])
                if list2[j] not in eitherOr:
                    eitherOr.append(list2[j])

#remove students who are in both classes from eitherOr
    for i in range(0, len(bothClasses)):
                eitherOr.remove(bothClasses[i])
#Print lists
    print('Students in both classes:', bothClasses)
    print('Students in either Python or Web Application:', eitherOr)

Python = ['Mr.Pickles','Jeff','Jennifer','Brian']

Web_Application = ['Mr.Pickles','What','TestSubject','Brian','hehexd']

sepLists(Python, Web_Application)

