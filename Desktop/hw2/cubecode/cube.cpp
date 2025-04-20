// rubiks_cube.cpp
#include "../include/glad.h"
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <vector>

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
    glm::vec3 position;
    int id;
};

std::vector<Cubelet> cubelets;

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        rotateX -= 0.02f;
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        rotateX += 0.02f;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        rotateY -= 0.02f;
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        rotateY += 0.02f;
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
    // Generate cubelets
    float offset = 0.7f;
    int id = 0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            for (int z = -1; z <= 1; ++z) {
                Cubelet c;
                c.position = glm::vec3(x * offset, y * offset, z * offset);
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
    createCube(baseVertices, baseIndices, glm::vec3(0.0f)); // centered at origin

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

    while (!glfwWindowShouldClose(window)) {
        processInput(window);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(shaderProgram);

        glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)SCR_WIDTH / SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = glm::lookAt(
            glm::vec3(4.0f, 3.0f, 6.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        view = glm::rotate(view, rotateX, glm::vec3(1.0f, 0.0f, 0.0f));
        view = glm::rotate(view, rotateY, glm::vec3(0.0f, 1.0f, 0.0f));

        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));

        glBindVertexArray(VAO);

        for (const Cubelet& c : cubelets) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, c.position);
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
