'''
Write a python program to create any one of the following management systems.
You can also pick one of your own.
a.Library Management System (should have classes for Person, Student, Librarian, Book etc.)
b.Airline Booking Reservation System (classes for Flight,Person,Employee,Passenger etc.)
c.Hotel Reservation System (classes for Room,Occupants,Employee etc.)
d.Student Enrollment System (classes for Student,System,Grades etc.)
e.Expense Tracker System (classes for Expense, Transaction Category etc.)

Prerequisites:
Your code should have at least five classes. xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Your code should have _init_ constructor in all the classes xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Your code should show inheritance at least once xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Your code should have one super call xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Use of self is required xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Use at least one private data member in your code. xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
Use multiple Inheritance at least once xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Create instances of all classes and show the relationship between them.
Your submission code should point out where all these things are present.
'''

#I chose:
#  a.Library Management System

'''
Classes:
1.Library
2.Person
3.Book
4.Student
5.Employee 
6.Librarian
'''

#Class 1
class Library(object): #super class
    checkedOut = []
    books = []
    people = []
    def __init__(self, state, city, country, address, zip, libName): #init constructors uses self
        self.state = state
        self.city = city
        self.country = country
        self.address = address
        self.zip = zip
        self.libName = libName

#add, remove, print books in library
    def add_book(self, book):
        if book not in self.books:
            self.books.append(book)

    def remove_book(self, book):
        if book in self.books:
            self.books.remove(book)

    def print_books(self):
        for book in self.books:
            print(book.title)

#add, remove, print people/customers/employees... in library
    def add_person(self, person):
        if person not in self.people:
            self.people.append(person)

    def remove_person(self, person):
        if person in self.people:
            self.people.remove(person)

    def print_people(self):
        for person in self.people:
            print(person.firstName)

#check out, return books, and print books checked out
    def checkOut(self, book_title):
        if self.books != []:
            for i in range(0, len(self.books)):
                if(self.books[i].title == book_title and self.books[i].available != 0):
                    self.checkedOut.append(self.books[i])
                    self.books[i].available = self.books[i].available - 1
                    print('book checked out')
        else:
            print('There are no books')

    def return_book(self, book_title):
        if self.checkedOut != []:
            for i in range(0, len(self.checkedOut)-1):
                save = self.checkedOut[i].title
                if type(save) == type(book_title):
                    del self.checkedOut[i]
            print('book returned')
        else:
            print('No books checked Out')

        for i in range(0, len(self.books)):
            if (self.books[i].title == book_title):
                self.books[i].available = self.books[i].available + 1

    def print_checkedOut(self):
        for chkt in self.checkedOut:
            print(chkt.title)
#Class 2
class Person(Library):
    def __init__(self, firstName, lastName, bookCheckedOut, fine, state, city, country, address, zip, libName): #init constructors uses self
        super().__init__(state, city, country, address, zip, libName)
        self.firstName = firstName
        self.lastName = lastName
        self.booksCheckedOut = bookCheckedOut
        self.fine = fine


#Class 3
class Book():
    def __init__(self, title, author, datePublished, dateCheckedOut, dateReturn, waitList, cost, topic, available): #init constructors uses self
        self.title = title
        self.author = author
        self.datePublished = datePublished
        self.dateCheckedOut = dateCheckedOut
        self.dateReturn = dateReturn
        self.waitList = waitList
        self.cost = cost
        self.topic = topic
        self.available = available

    def book_title(self):
        return self.title

#Class 4
class Student(Person): #inheritance
    def __init__(self, studentID, studentSchool, degreeProgram, firstName, lastName, bookCheckedOut, fine,state, city, country, address, zip, libName): #init constructors uses self
        super().__init__(firstName, lastName, bookCheckedOut, fine, state, city, country, address, zip, libName) #super class
        self.__studentID = studentID #private data member
        self.studentSchool = studentSchool
        self.degreeProgram = degreeProgram
#Class 5
class employee:
    def __init__(self, employeeID, position): #init constructors
        self.__employeeID = employeeID #private data member
        self.position = position

#Class 6
class Librarian(Person, employee): #multiple inheritance
    def __init__(self, employeeID, position,firstName, lastName, bookCheckedOut, fine, state, city, country, address, zip, libName): #init constructors uses self
        employee.__init__(self, employeeID, position)
        Person.__init__(self, firstName, lastName, bookCheckedOut, fine, state, city, country, address, zip, libName) #super class
        self.pay = 30000

#Create instances of all classes and show the relationship between them.
#Library Class - create the Library
#Library is the super class for Person
Library = Library('Missouri', 'Kansas City', 'USA', '11111 imaginary street', 111111, 'The Library')

#Book Class - create a book object
Book1 = Book('The Art of War', 'Sun Zu', '1/1/1700', '2/15/2018', '2/16/2018', 1, 10, 'Strategy', 4)
Book2 = Book('Of Mice and Men', 'Sun Zu', '1/1/1700', '2/15/2018', '2/16/2018', 1, 10, 'Strategy', 1)
Book3 = Book('How to be a student', 'Sun Zu', '1/1/1700', '2/15/2018', '2/16/2018', 1, 10, 'Strategy',0)

#Stock Library with books, one of the books has an available value of 0 therefore it cannot be checked out
Library.add_book(Book1)
Library.add_book(Book2)
Library.add_book(Book3)

#Print book available in the library
Library.print_books()
print('----------------------------------------------------------')

#Person Class & Employee class used for student and librarian to inherit from
# where Librarian(Person, Employee) - multiple inheritance
#  and Student(Person) - inheritance
'''Create Student'''
Student1 = Student(111111, 'UMKC', 'ECE', 'test', 'student', 'The Art of War', 0,'Missouri', 'Kansas City', 'USA', '11111 imaginary street', 111111, 'The Library')
#The student can checkout and return books because it inherits Persons attributes, which is connected to the super class Library
print('-------------------------Student------------------------------')
Student1.checkOut('The Art of War')
Student1.return_book('The Art of War')
print('-------------------------------------------------------')

'''Create Librarian'''
librian1 = Librarian(123, 'shelf', 'test', 'employee', 'NA', 0,'Missouri', 'Kansas City', 'USA', '11111 imaginary street', 111111, 'The Library')
#The Librarian can checkout and return books because it inherits Persons attributes, which is connected to the super class Library
#The Librarian also inherits the employee id and position from the employee class
print('------------------------Librarian-------------------------------')
librian1.checkOut('The Art of War')
librian1.checkOut('How to be a student') #book has an available value of 0 therefore it cannot be checked out
librian1.return_book('The Art of War')
print('-------------------------------------------------------')

#This line shows that the employeeID is private because we cannot access it (uncomment to see)


#This line shows that the librarian also inherits from employee
print('Librarian Position: ',librian1.position)

#This is here because it says to make instances of all classes
employee1 = employee(123, 'manager')
person = Person('firstName', 'lastName', 'bookCheckedOut', 0, 'Missouri', 'Kansas City', 'USA', '11111 imaginary street', 111111, 'The Library')
print("this should give error because it is private")
print(librian1.employeeID)
