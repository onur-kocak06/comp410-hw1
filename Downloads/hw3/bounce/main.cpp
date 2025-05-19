#include "../include/Angel.h"
#include <glm/gtc/type_ptr.hpp>

const int WIDTH = 800, HEIGHT = 600;

// Position and movement
float posX = -0.9f, posY = 0.9f;
float velocityX = 0.005f, velocityY = 0.0f;
float gravity = -0.001f;
float damping = 0.9f;

// Rendering mode
bool isWireframe = false;

// Colors
bool isRed = true;
glm::vec3 red = glm::vec3(1.0f, 0.0f, 0.0f);
glm::vec3 green = glm::vec3(0.0f, 1.0f, 0.0f);

// Key press states to avoid repeats
static bool sKeyPressed = false;
static bool oKeyPressed = false;
static bool lKeyPressed = false;
static bool mKeyPressed = false;
static bool zKeyPressed = false;
static bool wKeyPressed = false;
static bool tKeyPressed = false;
static bool iKeyPressed = false;

// Shading and lighting toggles
bool useGouraud = false;
bool toggleAmbient = true;
bool toggleDiffuse = true;
bool toggleSpecular = true;
bool useMetallic = false;
bool lightAttachedToObject = false;

// Zoom factor
float zoom = 1.0f;

// Shader and buffers
unsigned int shaderProgram;
unsigned int sphereVAO = 0, sphereVBO = 0, sphereEBO = 0;
unsigned int indexCount = 0;

const char* vertexShaderSource = R"glsl(
#version 330 core

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 texCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec3 lightDirection;
uniform float shininess;
uniform bool toggleAmbient;
uniform bool toggleDiffuse;
uniform bool toggleSpecular;
uniform bool useGouraud;

out vec3 FragPos;
out vec3 Normal;
out vec2 TexCoord;
out vec3 gouraudColor;

void main()
{
    FragPos = vec3(model * vec4(position, 1.0));
    Normal = mat3(transpose(inverse(model))) * normal;
    TexCoord = texCoord;

    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(-lightDirection);
    vec3 viewDir = normalize(-FragPos);  // camera at origin

    vec3 ambient = toggleAmbient ? vec3(0.1) : vec3(0.0);
    vec3 diffuse = toggleDiffuse ? max(dot(norm, lightDir), 0.0) * vec3(0.6) : vec3(0.0);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = toggleSpecular ? pow(max(dot(viewDir, reflectDir), 0.0), shininess) : 0.0;
    vec3 specular = spec * vec3(0.8);

    gouraudColor = ambient + diffuse + specular;

    gl_Position = projection * view * model * vec4(position, 1.0);
}
)glsl";

const char* fragmentShaderSource = R"glsl(
#version 330 core

in vec3 FragPos;
in vec3 Normal;
in vec2 TexCoord;
in vec3 gouraudColor;

out vec4 FragColor;

uniform sampler2D texture1;
uniform vec3 lightDirection;
uniform float shininess;
uniform bool toggleAmbient;
uniform bool toggleDiffuse;
uniform bool toggleSpecular;
uniform bool useGouraud;
uniform bool useTexture;

void main()
{
    if (useGouraud) {
        FragColor = vec4(gouraudColor, 1.0);
    } else {
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(-lightDirection);
        vec3 viewDir = normalize(-FragPos);

        vec3 ambient = toggleAmbient ? vec3(0.1) : vec3(0.0);
        vec3 diffuse = toggleDiffuse ? max(dot(norm, lightDir), 0.0) * vec3(0.6) : vec3(0.0);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = toggleSpecular ? pow(max(dot(viewDir, reflectDir), 0.0), shininess) : 0.0;
        vec3 specular = spec * vec3(0.8);

        vec3 result = ambient + diffuse + specular;
        vec3 baseColor = useTexture ? texture(texture1, TexCoord).rgb : vec3(1.0);
        FragColor = vec4(baseColor * result, 1.0);
    }
}
)glsl";

// Function declarations
void createShader();
void generateSphere(float radius, unsigned int sectorCount, unsigned int stackCount);
void drawSphere(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection);
void processInput(GLFWwindow* window);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "3D Sphere", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    // Set callbacks
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    // Load GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return -1;
    }

    glViewport(0, 0, WIDTH, HEIGHT);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    createShader();
    generateSphere(0.5f, 40, 40);

    // Setup view and projection matrices
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), float(WIDTH)/HEIGHT, 0.1f, 100.0f);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        velocityY += gravity;
        //posY += velocityY;
        //posX += velocityX;

        // Bounce on bottom boundary
        if (posY <= -0.9f) {
            posY = -0.9f;
            velocityY *= -damping;
        }

        glClearColor(0.2f, 0.2f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera/view matrix (fixed camera at z=3 looking at origin)
        glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 3 / zoom), glm::vec3(0,0,0), glm::vec3(0,1,0));

        // Model matrix with translation based on posX and posY
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(posX, posY, 0.0f));

        // Use shader program
        glUseProgram(shaderProgram);

        // Set uniforms
        glUniform1i(glGetUniformLocation(shaderProgram, "useGouraud"), useGouraud ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleAmbient"), toggleAmbient ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleDiffuse"), toggleDiffuse ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleSpecular"), toggleSpecular ? 1 : 0);
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), useMetallic ? 128.0f : 32.0f);
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), 0); // no texture by default

        // Light direction uniform
        glm::vec3 lightDir = lightAttachedToObject ? glm::vec3(model * glm::vec4(0.0, 0.0, 1.0, 0.0)) : glm::vec3(0.0, 0.0, 1.0);
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightDirection"), 1, glm::value_ptr(lightDir));


    // Set matrices
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
    glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

    // Draw sphere
    drawSphere(model, view, projection);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

glfwDestroyWindow(window);
glfwTerminate();
return 0;
}

void createShader() {
unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
glCompileShader(vertexShader);
//check_shader_compile(vertexShader);

unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
glCompileShader(fragmentShader);
//check_shader_compile(fragmentShader);

shaderProgram = glCreateProgram();
glAttachShader(shaderProgram, vertexShader);
glAttachShader(shaderProgram, fragmentShader);
glLinkProgram(shaderProgram);
//check_program_link(shaderProgram);

glDeleteShader(vertexShader);
glDeleteShader(fragmentShader);
}

void generateSphere(float radius, unsigned int sectorCount, unsigned int stackCount) {
std::vector<float> vertices;
std::vector<unsigned int> indices;

for (unsigned int i = 0; i <= stackCount; ++i) {
    float stackAngle = glm::pi<float>() / 2 - i * glm::pi<float>() / stackCount;
    float xy = radius * cosf(stackAngle);
    float z = radius * sinf(stackAngle);

    for (unsigned int j = 0; j <= sectorCount; ++j) {
        float sectorAngle = j * 2 * glm::pi<float>() / sectorCount;
        float x = xy * cosf(sectorAngle);
        float y = xy * sinf(sectorAngle);

        float nx = x / radius, ny = y / radius, nz = z / radius;
        float s = (float)j / sectorCount;
        float t = (float)i / stackCount;

        vertices.insert(vertices.end(), {x, y, z, nx, ny, nz, s, t});
    }
}

for (unsigned int i = 0; i < stackCount; ++i) {
    for (unsigned int j = 0; j < sectorCount; ++j) {
        unsigned int k1 = i * (sectorCount + 1) + j;
        unsigned int k2 = k1 + sectorCount + 1;

        indices.insert(indices.end(), {k1, k2, k1+1, k1+1, k2, k2+1});
    }
}

indexCount = indices.size();

glGenVertexArrays(1, &sphereVAO);
glGenBuffers(1, &sphereVBO);
glGenBuffers(1, &sphereEBO);

glBindVertexArray(sphereVAO);
glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

GLsizei stride = 8 * sizeof(float);
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, (void*)0); // position
glEnableVertexAttribArray(0);
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, (void*)(3 * sizeof(float))); // normal
glEnableVertexAttribArray(1);
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, stride, (void*)(6 * sizeof(float))); // texCoord
glEnableVertexAttribArray(2);

glBindVertexArray(0);
}

void drawSphere(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection) {
glBindVertexArray(sphereVAO);
glPolygonMode(GL_FRONT_AND_BACK, isWireframe ? GL_LINE : GL_FILL);
glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
glBindVertexArray(0);
}

// Mouse click: toggle wireframe
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
isWireframe = !isWireframe;
}

// Scroll: zoom
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
zoom *= (1.0f + yoffset * 0.1f);
if (zoom < 0.1f) zoom = 0.1f;
if (zoom > 10.0f) zoom = 10.0f;
}

// Keyboard input
void processInput(GLFWwindow* window) {
if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
glfwSetWindowShouldClose(window, true);

if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    // Toggle shading mode: Gouraud / Phong
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !sKeyPressed) {
        useGouraud = !useGouraud;
        sKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_RELEASE)
        sKeyPressed = false;

    // Toggle lighting components (ambient, diffuse, specular)
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !oKeyPressed) {
        if (toggleAmbient && toggleDiffuse && toggleSpecular)
            toggleAmbient = false;
        else if (!toggleAmbient && toggleDiffuse && toggleSpecular)
            toggleDiffuse = false;
        else if (!toggleAmbient && !toggleDiffuse && toggleSpecular)
            toggleSpecular = false;
        else
            toggleAmbient = toggleDiffuse = toggleSpecular = true;
        oKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_RELEASE)
        oKeyPressed = false;

    // Toggle light mode: fixed or object-bound
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS && !lKeyPressed) {
        lightAttachedToObject = !lightAttachedToObject;
        lKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_RELEASE)
        lKeyPressed = false;

    // Toggle material type: plastic or metallic
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !mKeyPressed) {
        useMetallic = !useMetallic;
        mKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_RELEASE)
        mKeyPressed = false;

    // Zoom in
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS && !zKeyPressed) {
        zoom *= 1.1f;
        if (zoom > 10.0f) zoom = 10.0f;
        zKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_Z) == GLFW_RELEASE)
        zKeyPressed = false;

    // Zoom out
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS && !wKeyPressed) {
        zoom *= 0.9f;
        if (zoom < 0.1f) zoom = 0.1f;
        wKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_RELEASE)
        wKeyPressed = false;

    // Toggle rendering mode (e.g., shaded, wireframe, textured)
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_PRESS && !tKeyPressed) {
        isWireframe = !isWireframe;
        tKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE)
        tKeyPressed = false;

    // Toggle between different texture images
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !iKeyPressed) {
        //currentTextureIndex = (currentTextureIndex + 1) % totalTextureCount;
        //updateTexture(currentTextureIndex); // You should define this
        iKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_RELEASE)
        iKeyPressed = false;
}