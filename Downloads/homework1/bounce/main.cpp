#include "../include/Angel.h"

// Window settings
const int WIDTH = 800, HEIGHT = 600;

// Ball properties
float posX = -0.9f, posY = 0.9f; // Start at top-left
float velocityX = 0.005f, velocityY = 0.0f; // initial velocity
float gravity = -0.001f;
float damping = 0.9f; // reduce for a less bouncy ball
bool isSphere = false;
bool isWireframe = false;

// Ball color
bool isRed = true;
glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
glm::vec3 green = glm::vec3(0.0f, 1.0f, 0.0f);

// Flags for each key press
bool qKeyPressed = false;
bool iKeyPressed = false;
bool cKeyPressed = false;
bool hKeyPressed = false;

// OpenGL shaders
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
uniform mat4 transform;
void main() {
    gl_Position = transform * vec4(aPos, 1.0);
})";

const char* fragmentShaderSource = R"(
#version 330 core
out vec4 FragColor;
uniform vec3 color;
void main() {
    FragColor = vec4(color, 1.0);
})";

unsigned int shaderProgram;
void createShader();
void processInput(GLFWwindow* window);
void drawObject();
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "Bouncing Ball", NULL, NULL);
    if (!window) { std::cout << "Failed to create GLFW window"; return -1; }
    glfwMakeContextCurrent(window);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) { return -1; }
    glViewport(0, 0, WIDTH, HEIGHT);

    createShader();

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Physics update
        velocityY += gravity;
        posY += velocityY;
        posX += velocityX;

        // Bounce when hitting the floor
        if (posY <= -0.9f) {
            posY = -0.9f;
            velocityY *= -damping;
        }


        glClear(GL_COLOR_BUFFER_BIT);
        drawObject();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}

void createShader() {
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

// Function declarations
void drawCube();
void drawSphere();
std::vector<float> generateSphereVertices(float radius, unsigned int sectorCount, unsigned int stackCount);

void drawObject() {
    glUseProgram(shaderProgram);

    if (isSphere) {
        drawSphere();
    } else {
        drawCube();
    }
}

void drawCube() {
    float vertices[] = {
        -0.05f, -0.05f, 0.0f,
         0.05f, -0.05f, 0.0f,
         0.05f,  0.05f, 0.0f,
        -0.05f,  0.05f, 0.0f
    };
    unsigned int indices[] = {0, 1, 2, 2, 3, 0};

    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(posX, posY, 0.0f));
    unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, &transform[0][0]);

    unsigned int colorLoc = glGetUniformLocation(shaderProgram, "color");
    if(isRed){
        glUniform3f(colorLoc, red.r, red.g, red.b);
    }
    else{
        glUniform3f(colorLoc, green.r, green.g, green.b);
    }

    glPolygonMode(GL_FRONT_AND_BACK, isWireframe ? GL_LINE : GL_FILL);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindVertexArray(0);
}

void drawSphere() {
    std::vector<float> vertices = generateSphereVertices(0.1f, 50, 50);  // Sphere with 50 stacks and sectors

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glm::mat4 transform = glm::translate(glm::mat4(1.0f), glm::vec3(posX, posY, 0.0f));
    unsigned int transformLoc = glGetUniformLocation(shaderProgram, "transform");
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, &transform[0][0]);

    unsigned int colorLoc = glGetUniformLocation(shaderProgram, "color");
    if(isRed){
        glUniform3f(colorLoc, red.r, red.g, red.b);
    }
    else{
        glUniform3f(colorLoc, green.r, green.g, green.b);
    }

    glPolygonMode(GL_FRONT_AND_BACK, isWireframe ? GL_LINE : GL_FILL);
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 3);

    glBindVertexArray(0);
}

std::vector<float> generateSphereVertices(float radius, unsigned int sectorCount, unsigned int stackCount) {
    std::vector<float> vertices;
    float x, y, z, xy;                              // Vertex position
    float sectorStep = 2 * M_PI / sectorCount;
    float stackStep = M_PI / stackCount;
    float sectorAngle, stackAngle;

    // Loop through each stack and sector to create vertices
    for (unsigned int i = 0; i <= stackCount; ++i) {
        stackAngle = M_PI / 2 - i * stackStep;         // Starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);                // xy projection of radius
        z = radius * sinf(stackAngle);                 // z = radius * sin(stackAngle)

        // Loop through each sector
        for (unsigned int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;              // Angle around the circle
            x = xy * cosf(sectorAngle);                // X = radius * cos(stackAngle) * cos(sectorAngle)
            y = xy * sinf(sectorAngle);                // Y = radius * cos(stackAngle) * sin(sectorAngle)

            // Add the vertex
            vertices.push_back(x);
            vertices.push_back(y);
            vertices.push_back(z);
        }
    }
    return vertices;
}

void processInput(GLFWwindow* window) {
    // Handle the "Q" key press
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS && !qKeyPressed) {
        glfwSetWindowShouldClose(window, true);
        qKeyPressed = true; // Prevent multiple actions for the same key press
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_RELEASE) {
        qKeyPressed = false; // Reset flag when the key is released
    }

    // Handle the "I" key press
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !iKeyPressed) {
        posX = -0.9f; 
        posY = 0.9f; 
        velocityY = 0.0f;
        iKeyPressed = true; // Prevent multiple actions for the same key press
    }
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_RELEASE) {
        iKeyPressed = false; // Reset flag when the key is released
    }

    // Handle the "C" key press
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS && !cKeyPressed) {
        isRed = !isRed;
        cKeyPressed = true; // Prevent multiple actions for the same key press
    }
    if (glfwGetKey(window, GLFW_KEY_C) == GLFW_RELEASE) {
        cKeyPressed = false; // Reset flag when the key is released
    }

    // Handle the "H" key press
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS && !hKeyPressed) {
        std::cout << "Press i to reset, c to change color, q to quit, right-click to switch object, left-click for wireframe.\n";
        hKeyPressed = true; // Prevent multiple actions for the same key press
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_RELEASE) {
        hKeyPressed = false; // Reset flag when the key is released
    }
}
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) isSphere = !isSphere;
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) isWireframe = !isWireframe;
}
