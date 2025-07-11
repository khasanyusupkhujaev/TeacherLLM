# from google import genai

# client = genai.Client(api_key="AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ")
# chat = client.chats.create(model="gemini-2.0-flash")

# response = chat.send_message('''
# You are a mathematics grading assistant. Grade the following solution step-by-step.
#     Provide detailed feedback, identify errors, and award partial credit where applicable. \n\
#     Problem: Determine the equation of the plane passing through the points A(1, 2, 3), B(4, 5, 6), and C(7, 8, 9). 
# Additionally, find the distance from the plane to the point P(10, 11, 12).

# Student Solution:
# Step 1: Find two vectors on the plane:
#   Vector AB = B - A = (4 - 1, 5 - 2, 6 - 3) = (3, 3, 3)
#   Vector AC = C - A = (7 - 1, 8 - 2, 9 - 3) = (6, 6, 6)

# Step 2: Compute the normal vector to the plane:
#   Normal vector = AB × AC = 
#   |  i   j   k  |
#   |  3   3   3  |
#   |  6   6   6  |
#   = i(3*6 - 3*6) - j(3*6 - 3*6) + k(3*6 - 3*6)
#   = (0, 0, 0).

# Step 3: Write the equation of the plane:
#   Using point A(1, 2, 3) and the normal vector (0, 0, 0),
#   The equation of the plane is 0x + 0y + 0z = 0.

# Step 4: Find the distance from the point P(10, 11, 12) to the plane:
#   Distance = |Ax + By + Cz + D| / sqrt(A^2 + B^2 + C^2)
#   Substituting A = 0, B = 0, C = 0, D = 0, and P = (10, 11, 12):
#   Distance = |0(10) + 0(11) + 0(12) + 0| / sqrt(0^2 + 0^2 + 0^2)
#   Distance = undefined.
# Provide your feedback and conclude with a final grade.
#                              ''')
# print(response.text)

# for message in chat.get_history():
#     print(f'role - {message.role}',end=": ")
#     print(message.parts[0].text)


from google import genai
from google.genai import types

# Path to the image file
image_path = "example3.jpg"

# Read the image as bytes
with open(image_path, "rb") as img_file:
    image_bytes = img_file.read()

# Prepare the image part
image = types.Part.from_bytes(
    data=image_bytes,
    mime_type="image/jpeg"
)

# Initialize the Gemini client
client = genai.Client(api_key="AIzaSyAL2dFskGajHSf0N3LbYsClqs_9LvAc3tQ")

# Generate content
response = client.models.generate_content(
    model="gemini-2.0-flash-exp",
    contents=["I want you to return what is written in the image. if there is a graphical represintation, just explain it with words.", image],
)

# Print the response
print(response.text)
