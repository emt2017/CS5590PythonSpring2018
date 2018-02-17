'''
With any given number n,  In any mobile , there is contact list.
Create a list of contacts and then prompt the user to do the following:
 a)Display contact by name
 b)Display contact by number
 c)Edit contact by name
 d)Exit
 Based on the above scenario, write a single program to perform the above the operations.
 Each time an operation is performed on the list, the contact list should be displayed
 Sample input:
 Contact_list=[{“name”:”Rashmi”,”number”:”8797989821”,”email”:”rr@gmail.com”},{ “name”:”Saria”,”number”:”9897989821”,”email”:”ss@gmail.com”}]
 Suppose user select to edit contact “Rashmi”Edit the number to 999999999 as given by user
 Sample output:
 Contact_list=[{“name”:”Rashmi”,”number   ”:”9999999999”,”email  ”:”rr@gmail.com”},{ “name”:”Saria”,”number   ”:”9897989821”,”email  ”:”ss@gmail.com”
'''
import string


def displayByName(name, Contact_List):
    display = []
    for i in Contact_List: #find index in contactlist where given name is
        if i["name"] == name:#if index is in contactlist append to the display
            display.append(i)
    if display != []: #check display isn't empty before printing it
        print(display)
        print('')
    else: #else if display is empty warn user could not find books
        print('Sorry, that name is not in the books')

def displayByNumber(number, Contact_List):
    display = []
    for i in Contact_List:#find index in contactlist where given number is
        if i["number"] == number:#if index is in contactlist append to the display
            display.append(i)
    if display != []:#check display isn't empty before printing it
        print(display)
        print('')
    else:#else if display is empty warn user could not find books
        print('Sorry, that number is not in the books')

def editContactByName(name, Contact_List):
    customerInput = ''
    for i in Contact_List:
        if i["name"] == name:
            while customerInput != 'd':

                print('a)Edit name')
                print('b)Edit number')
                print('c)Edit email')
                print('d)Back to Main Menu')

                customerInput = input('Enter your option: ')
                if customerInput == 'a' or customerInput == 'b' or customerInput == 'c':
                    if customerInput == 'a':
                        editName = input('Please enter the new name: ').strip() #get new name and strip leading and tailing whitespace
                        i["name"] = editName

                    elif customerInput == 'b':
                        # get new number and strip leading and tailing whitespace, parenthesis, and dashes
                        editNumber = input('Please enter the new number: ').replace(" ","").replace("-","").replace("(","").replace(")","")
                        #check if all numbers
                        if editNumber.isdigit() and (len(editNumber) == 10 or len(editNumber) == 7):
                            i["number"] = editNumber
                        else:
                            print("Please enter a valid phone number")

                    elif customerInput == 'c':
                        editEmail = input('Please enter the new email: ').replace(" ","")
                        #check if @ and .
                        if '@' in editEmail and '.' in editEmail:
                            i["email"] = editEmail
                        else:
                            print("Please enter a valid email")

                elif customerInput != 'a' and customerInput != 'b' and customerInput != 'c' and customerInput != 'd':
                    print('Please enter a valid option')
                elif customerInput == 'd':
                    print('Main Menu')


#Variables
customerInput = ''
Contact_list = [{'name':'Rashmi','number':'8797989821','email':'rr@gmail.com'},{ 'name':'Saria','number':'9897989821','email':'ss@gmail.com'}]

while customerInput != 'd':

    print('a)Display contact by name')
    print('b)Display contact by number')
    print('c)Edit contact by name')
    print('d)Exit')

    customerInput = input('Enter your option: ')
    if customerInput == 'a' or customerInput == 'b' or customerInput == 'c':
        if customerInput == 'a':
            name = ''
            name = input('Please input a name to search: ')
            displayByName(name, Contact_list)

        elif customerInput =='b':
            number = ''
            number = input('Please input a number to search: ')
            displayByNumber(number, Contact_list)

        elif customerInput =='c':
            name = ''
            name = input('Please input a name to search: ')
            editContactByName(name, Contact_list)

    elif customerInput != 'a' and customerInput != 'b' and customerInput != 'c' and customerInput != 'd':
        print('Please enter a valid option')
    elif customerInput == 'd':
        print('The program will now exit')
