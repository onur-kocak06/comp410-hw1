// rubiks_cube.cpp
#include "../include/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>

const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;
float rotateX = 0.0f;
float rotateY = 0.0f;

const char* vertexShaderSource = R"glsl(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


out vec3 ourColor;

void main() {
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    ourColor = aColor;
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;

void main() {
    FragColor = vec4(ourColor, 1.0);
}
)glsl";

struct Cubelet {
    glm::vec3 originalPos;
    glm::mat4 transform = glm::mat4(1.0f);
    int id;
};


bool hKeyPressed = false;

std::vector<Cubelet> cubelets;
glm::vec3 activeRotationAxis;
float activeRotationDirection = 1.0f;

// Animation state
bool isRotating = false;
float currentAngle = 0.0f;
const float targetAngle = glm::radians(90.0f);
const float rotationSpeed = glm::radians(180.0f); // degrees per second
std::vector<int> rotatingIds;
glm::vec3 activePivot = glm::vec3(0.0f);


void startRotation(char axis, float coord, bool clockwise) {
    if (isRotating) return;

    rotatingIds.clear();
    glm::vec3 rotAxis;

    // Determine the rotation axis (perpendicular to the slice you're rotating)
    switch (axis) {
        case 'x': rotAxis = glm::vec3(1, 0, 0); break; // rotating left/right column -> around Y
        case 'y': rotAxis = glm::vec3(0, 1, 0); break; // rotating top/bottom row -> around X
        case 'z': rotAxis = glm::vec3(0, 0, 1); break; // rotating front/back face -> around Z
        default: return; // invalid axis
    }
    // Determine center of rotation (pivot point)
    activePivot = glm::vec3(0.0f);
    switch (axis) {
        case 'x': activePivot.x = coord; break;
        case 'y': activePivot.y = coord; break;
        case 'z': activePivot.z = coord; break;
    }


    // Select the cubelets on the slice to rotate
    for (int i = 0; i < cubelets.size(); ++i) {
        glm::vec4 transformed = cubelets[i].transform * glm::vec4(cubelets[i].originalPos, 1.0f);
        glm::vec3 pos = glm::vec3(glm::round(glm::vec3(transformed) * 10.0f) / 10.0f);
        if (std::abs((axis == 'x' ? pos.x : axis == 'y' ? pos.y : pos.z) - coord) < 0.1f) {
            rotatingIds.push_back(i);
        }
    }

    currentAngle = 0.0f;
    isRotating = true;
    activeRotationAxis = rotAxis;
    activeRotationDirection = clockwise ? -1.0f : 1.0f;
}




void updateRotation(float dt) {
    // advance the animation
    float step = rotationSpeed * dt;
    if (currentAngle + step >= targetAngle) {
        step = targetAngle - currentAngle;
        isRotating = false;
    }
    currentAngle += step;

    // build the tiny delta‐step around the correct axis
    glm::mat4 rotStep = glm::rotate(glm::mat4(1), step * activeRotationDirection, activeRotationAxis);

    // apply it to each affected cubelet
    for (int id : rotatingIds) {
        Cubelet& c = cubelets[id];
        c.transform = glm::translate(glm::mat4(1), activePivot) 
                    * rotStep 
                    * glm::translate(glm::mat4(1), -activePivot) 
                    * c.transform;
    }

    // once we hit 90°, finalize:
    if (!isRotating) {
        
        rotatingIds.clear();
        return;
    }
    
}
void printUseInstractions(){
    printf("use the arrow keys to ratate the cube\n");
    printf("press \'W\' to rotate the top horizontal node\n");
    printf("press \'S\' to rotate the bottom horizontal node\n");
    printf("press \'A\' to rotate the left vertical node\n");
    printf("press \'D\' to rotate the right vertical node\n");
    printf("press \'R\' to randomize the cube\n");
    printf("press \'Q\' to quit\n");
}


void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotateX -= 0.02f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        rotateX += 0.02f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        rotateY -= 0.02f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        rotateY += 0.02f;
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !isRotating)
        startRotation('y', -0.7f, true);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS && !isRotating)
        startRotation('y', 0.7f, true);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS && !isRotating)
        startRotation('x', 0.7f, true);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS && !isRotating)
        startRotation('x', -0.7f, true);
        if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS) {
            if (!hKeyPressed && !isRotating) {
                printUseInstractions();
                hKeyPressed = true;
            }
        } else {
            hKeyPressed = false; // Reset when key is released
        }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !isRotating)
        printf("i am suposed to randomize"); //TODO: implement randomizer
}

unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    return shader;
}

unsigned int createShaderProgram() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    return shaderProgram;
}

void createCube(std::vector<float>& vertices, std::vector<unsigned int>& indices, glm::vec3 offset) {
    float s = 0.3f;
    glm::vec3 positions[8] = {
        offset + glm::vec3(-s, -s,  s),
        offset + glm::vec3( s, -s,  s),
        offset + glm::vec3( s,  s,  s),
        offset + glm::vec3(-s,  s,  s),
        offset + glm::vec3(-s, -s, -s),
        offset + glm::vec3( s, -s, -s),
        offset + glm::vec3( s,  s, -s),
        offset + glm::vec3(-s,  s, -s)
    };

    glm::vec3 colors[6] = {
        {1.0f, 0.0f, 0.0f},  // front - red
        {1.0f, 0.5f, 0.0f},  // back - orange
        {0.0f, 0.0f, 1.0f},  // left - blue
        {0.0f, 1.0f, 0.0f},  // right - green
        {1.0f, 1.0f, 1.0f},  // top - white
        {1.0f, 1.0f, 0.0f}   // bottom - yellow
    };
    
    int faceIndices[6][4] = {
        {0, 1, 2, 3}, // front
        {5, 4, 7, 6}, // back
        {4, 0, 3, 7}, // left
        {1, 5, 6, 2}, // right
        {3, 2, 6, 7}, // top
        {4, 5, 1, 0}  // bottom
    };

    for (int f = 0; f < 6; ++f) {
        glm::vec3 color = colors[f];
        for (int i = 0; i < 4; ++i) {
            glm::vec3 pos = positions[faceIndices[f][i]];
            vertices.insert(vertices.end(), { pos.x, pos.y, pos.z, color.r, color.g, color.b });
        }
        unsigned start = vertices.size() / 6 - 4;
        indices.insert(indices.end(), {
            start, start + 1, start + 2,
            start, start + 2, start + 3
        });
    }
}

int main() {
    float offset = 0.7f;
    int id = 0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
                Cubelet c;
                c.originalPos = glm::vec3(x * offset, y * offset, z * offset);
                c.id = id++;
                cubelets.push_back(c);
            }
        }
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Rubik's Cube", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    unsigned int shaderProgram = createShaderProgram();

    std::vector<float> baseVertices;
    std::vector<unsigned int> baseIndices;
    createCube(baseVertices, baseIndices, glm::vec3(0.0f));

    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, baseVertices.size() * sizeof(float), baseVertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, baseIndices.size() * sizeof(unsigned int), baseIndices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    float lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float deltaTime = currentTime - lastTime;
        lastTime = currentTime;

        processInput(window);
        updateRotation(deltaTime);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(
            glm::vec3(4.0f, 3.0f, 6.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );
        view = glm::rotate(view, rotateX, glm::vec3(1, 0, 0));
        view = glm::rotate(view, rotateY, glm::vec3(0, 1, 0));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glBindVertexArray(VAO);

        for (const Cubelet& c : cubelets) {
            glm::mat4 model = glm::translate(glm::mat4(1.0f), c.originalPos);
            model = c.transform * model;
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            glDrawElements(GL_TRIANGLES, baseIndices.size(), GL_UNSIGNED_INT, 0);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glfwTerminate();
    return 0;
}
