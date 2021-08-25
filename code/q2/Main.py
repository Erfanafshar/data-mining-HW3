import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn import svm

linear = False
c_val = 1
kernel_type = 1
if kernel_type == 0:
    kernel_str = "rbf"
else:
    kernel_str = "poly"


def normalization(data, is_train):
    # 1.remove unimportant attributes
    # PassengerId
    data.drop("PassengerId", inplace=True, axis=1)

    # Name
    data.drop("Name", inplace=True, axis=1)
    # print(data.head())

    class_attribute = -1
    if is_train:
        # 2.copy class attribute then removing it from dataset
        # Survived
        class_attribute = data[["Survived"]].copy()
        data.drop("Survived", inplace=True, axis=1)
        # print(data.head())

    # 3.convert strings into numbers
    # Sex
    sex_attribute = data[["Sex"]].copy()
    sex_attribute = sex_attribute.values.tolist()
    data.drop("Sex", inplace=True, axis=1)
    sex_attribute_new = []
    for gender in sex_attribute:
        if gender == ["male"]:
            sex_attribute_new.append(1)
        else:
            sex_attribute_new.append(0)
    data["Sex"] = sex_attribute_new

    # Cabin
    cabin_attribute = data[["Cabin"]].copy()
    cabin_attribute = cabin_attribute.values.tolist()
    data.drop("Cabin", inplace=True, axis=1)
    cabin_attribute_new = []
    for cabin in cabin_attribute:
        if pd.isnull(cabin):
            cabin_attribute_new.append(0)
        else:
            cabin_attribute_new.append(1)
    data["Cabin"] = cabin_attribute_new

    # Embarked
    embarked_attribute = data[["Embarked"]].copy()
    embarked_attribute = embarked_attribute.values.tolist()
    data.drop("Embarked", inplace=True, axis=1)
    embarked_attribute_new = []
    for embarked in embarked_attribute:
        if pd.isnull(embarked):
            embarked_attribute_new.append(0)
        elif embarked == ["S"]:
            embarked_attribute_new.append(0)
        elif embarked == ["C"]:
            embarked_attribute_new.append(1)
        elif embarked == ["Q"]:
            embarked_attribute_new.append(2)
    data["Embarked"] = embarked_attribute_new
    # print(data.head())

    # 4.create intervals for numbers
    # Age
    age_attribute = data[["Age"]].copy()
    age_attribute = age_attribute.fillna(method='ffill')
    age_attribute = age_attribute.values.tolist()
    data.drop("Age", inplace=True, axis=1)
    age_attribute_new = []
    for age in age_attribute:
        if float(age[0]) <= 16:
            age_attribute_new.append(0)
        elif float(age[0]) <= 32:
            age_attribute_new.append(1)
        elif float(age[0]) <= 48:
            age_attribute_new.append(2)
        else:
            age_attribute_new.append(4)
    data["Age"] = age_attribute_new  #

    # Ticket
    # convert to number form
    ticket_attribute = data[["Ticket"]].copy()
    ticket_attribute = ticket_attribute.values.tolist()
    data.drop("Ticket", inplace=True, axis=1)
    ticket_attribute_new = []
    for ticket in ticket_attribute:
        # print(ticket[0])
        if " " in ticket[0]:
            parts = ticket[0].split(" ")
            if len(parts) == 2:
                ticket_number = parts[1]
            elif len(parts) == 3:
                ticket_number = parts[2]
            else:
                ticket_number = -1
                print("Error")
        else:
            if ticket[0] == "LINE":
                ticket_number = 10000
            else:
                ticket_number = ticket[0]
        ticket_attribute_new.append(int(ticket_number))

    # convert to interval form
    ticket_attribute_new_2 = []
    for num in ticket_attribute_new:
        if num <= 100E3:
            ticket_attribute_new_2.append(0)
        else:
            ticket_attribute_new_2.append(1)
    data["Ticket"] = ticket_attribute_new_2

    # Fare
    fare_attribute = data[["Fare"]].copy()
    fare_attribute = fare_attribute.values.tolist()
    data.drop("Fare", inplace=True, axis=1)
    fare_attribute_new = []
    for fare in fare_attribute:
        if float(fare[0]) <= 15:
            fare_attribute_new.append(0)
        elif float(fare[0]) <= 50:
            fare_attribute_new.append(1)
        else:
            fare_attribute_new.append(2)
    data["Fare"] = fare_attribute_new

    # Sibsp & Parch
    sibsp_attribute = data[["SibSp"]].copy()
    sibsp_attribute = sibsp_attribute.values.tolist()
    data.drop("SibSp", inplace=True, axis=1)

    parch_attribute = data[["Parch"]].copy()
    parch_attribute = parch_attribute.values.tolist()
    data.drop("Parch", inplace=True, axis=1)

    isAlone_attribute = []

    for i in range(len(sibsp_attribute)):
        if sibsp_attribute[i][0] > 0 or parch_attribute[i][0] > 0:
            isAlone_attribute.append(0)
        else:
            isAlone_attribute.append(1)
    # print(isAlone_attribute)
    data["isAlone"] = isAlone_attribute

    if is_train:
        return data, class_attribute
    else:
        return data


df = pd.read_csv("titanic\\train.csv")
df, ca = normalization(df, True)
X_train, X_test, y_train, y_test = train_test_split(df, ca, test_size=0.2, random_state=100)

if linear:
    clf = svm.LinearSVC(loss="hinge", C=c_val)
    clf.fit(X_train, y_train.values.ravel())
else:
    clf = svm.SVC(kernel=kernel_str, C=c_val, gamma="scale")
    clf.fit(X_train, y_train.values.ravel())

y_pred = clf.predict(X_test)
precision = precision_score(y_test, y_pred)
print("precision = ", precision)
y_test = y_test.values.ravel()
