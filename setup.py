
from setuptools import find_packages, setup # type: ignore
from typing import List


def get_requirements()->List[str]:
    
    #This function will return list of requirements
    
    requirements_lst:List[str]=[]
    try:
        with open('requirements.txt','r') as file:
            lines=file.readlines() #Read lines from the file
            #Process each line
            for line in lines:
                requirement=line.strip()
                #ignore empty lines and -e .
                if requirement and requirement!= '-e .':
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found")
    
    return requirements_lst

print(get_requirements())



setup(

    name="Customer_Risk_Escalation",
    version="0.0.1",
    author="Faizan Riaz",
    author_email="riazfaizan614@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
    

)