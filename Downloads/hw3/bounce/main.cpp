#include "../include/Angel.h"
#include <glm/gtc/type_ptr.hpp>

const int WIDTH = 800, HEIGHT = 600;

// Position and movement
float posX = -0.9f, posY = 0.9f;
float velocityX = 0.005f, velocityY = 0.0f;
float gravity = -0.001f;
float damping = 0.9f;
float velocityXsaved, velocityYsaved, gravitySaved = 0;
bool isPaused= false;
// Rendering mode
bool isWireframe = false;
enum RenderMode { WIREFRAME, SHADED, TEXTURED };
RenderMode renderMode = SHADED;

int currentTextureIndex = 0;

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
static bool rKeyPressed = false;
static bool pKeyPressed = false;
static bool hKeyPressed = false;

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

    if (useGouraud) {
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(-lightDirection);
        vec3 viewDir = normalize(-FragPos);  // camera at origin

        vec3 ambient = toggleAmbient ? vec3(0.1) : vec3(0.0);
        vec3 diffuse = toggleDiffuse ? max(dot(norm, lightDir), 0.0) * vec3(0.6) : vec3(0.0);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = toggleSpecular ? pow(max(dot(viewDir, reflectDir), 0.0), shininess) : 0.0;
        vec3 specular = spec * vec3(0.8);

        gouraudColor = ambient + diffuse + specular;
    }

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
        vec3 baseColor = useTexture ? texture(texture1, TexCoord).rgb : vec3(1.0);
        FragColor = vec4(baseColor * gouraudColor, 1.0);
    } else {
        vec3 norm = normalize(Normal);
        vec3 lightDir = normalize(-lightDirection);
        vec3 viewDir = normalize(-FragPos);

        vec3 ambient = toggleAmbient ? vec3(0.1) : vec3(0.0);
        vec3 diffuse = toggleDiffuse ? max(dot(norm, lightDir), 0.0) * vec3(0.6) : vec3(0.0);
        vec3 reflectDir = reflect(-lightDir, norm);
        float spec = toggleSpecular ? pow(max(dot(viewDir, reflectDir), 0.0), shininess) : 0.0;
        vec3 specular = spec * vec3(0.8);

        vec3 lighting = ambient + diffuse + specular;
        vec3 baseColor = useTexture ? texture(texture1, TexCoord).rgb : vec3(1.0);

        FragColor = vec4(baseColor * lighting, 1.0);
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
GLuint loadPPMTexture(const char* filename);

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

    // Compile and link shaders
    createShader();

    // Generate sphere mesh
    generateSphere(0.5f, 40, 40);

    // Load PPM textures
    GLuint texture1 = loadPPMTexture("basketball.ppm");
    GLuint texture2 = loadPPMTexture("earth.ppm");

    // Setup projection matrix
    glm::mat4 projection = glm::perspective(glm::radians(45.0f), float(WIDTH) / HEIGHT, 0.1f, 100.0f);

    while (!glfwWindowShouldClose(window)) {
        processInput(window);

        // Apply gravity and motion
        velocityY += gravity;
        posY += velocityY;
        posX += velocityX;

        if (posY <= -0.9f) {
            posY = -0.9f;
            velocityY *= -damping;
        }

        // Clear buffers
        glClearColor(0.2f, 0.2f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Camera view
        glm::mat4 view = glm::lookAt(glm::vec3(0, 0, 3.0f / zoom), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

        // Model transformation
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3(posX, posY, 0.0f));

        // Light direction
        glm::vec3 lightDir = lightAttachedToObject ? glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)) : glm::vec3(0.0f, -1.0f, 0.0f);

        // Activate shader
        glUseProgram(shaderProgram);

        // Set uniforms
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(shaderProgram, "lightDirection"), 1, glm::value_ptr(lightDir));
        glUniform1f(glGetUniformLocation(shaderProgram, "shininess"), useMetallic ? 16.0f : 4.0f);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleAmbient"), toggleAmbient ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleDiffuse"), toggleDiffuse ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "toggleSpecular"), toggleSpecular ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "useGouraud"), useGouraud ? 1 : 0);
        glUniform1i(glGetUniformLocation(shaderProgram, "useTexture"), renderMode == TEXTURED ? 1 : 0);

        // Bind appropriate texture if in textured mode
        if (renderMode == TEXTURED) {
            glActiveTexture(GL_TEXTURE0);
            GLuint texID = currentTextureIndex == 0 ? texture1 : texture2;
            glBindTexture(GL_TEXTURE_2D, texID);
            glUniform1i(glGetUniformLocation(shaderProgram, "texture1"), 0);
        }

        // Draw with correct mode
        if (renderMode == WIREFRAME) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        } else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

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
// layout: position (3), normal (3), texcoord (2)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
glEnableVertexAttribArray(0);

glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
glEnableVertexAttribArray(1);

glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
glEnableVertexAttribArray(2);


glBindVertexArray(0);
}

void drawSphere(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection) {
glBindVertexArray(sphereVAO);
glPolygonMode(GL_FRONT_AND_BACK, renderMode==WIREFRAME ? GL_LINE : GL_FILL);
glDrawElements(GL_TRIANGLES, indexCount, GL_UNSIGNED_INT, 0);
glBindVertexArray(0);
}


// Scroll: zoom
void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
zoom *= (1.0f + yoffset * 0.1f);
if (zoom < 0.1f) zoom = 0.1f;
if (zoom > 10.0f) zoom = 10.0f;
}

GLuint loadPPMTexture(const char* filename) {
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Failed to open " << filename << "\n";
        return 0;
    }

    std::string magic;
    file >> magic;
    if (magic != "P3") {
        std::cerr << "Invalid PPM format (must be ASCII P3): " << magic << "\n";
        return 0;
    }

    int width, height, maxval;
    file >> width >> height >> maxval;

    std::vector<unsigned char> data;
    data.reserve(width * height * 3);
    int r, g, b;
    while (file >> r >> g >> b) {
        data.push_back(static_cast<unsigned char>(r));
        data.push_back(static_cast<unsigned char>(g));
        data.push_back(static_cast<unsigned char>(b));
    }

    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    // Texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Upload to GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glGenerateMipmap(GL_TEXTURE_2D);

    return texture;
}



// Keyboard input
void processInput(GLFWwindow* window) {
if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

    // Toggle shading mode: Gouraud / Phong
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS && !sKeyPressed) {
        useGouraud = !useGouraud;
        std::cout << "useGouraud: " << (useGouraud ? "true" : "false") << "\n";
        sKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_RELEASE)
        sKeyPressed = false;

    // Toggle lighting components (ambient, diffuse, specular)
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS && !oKeyPressed) {
        if (toggleAmbient && toggleDiffuse && toggleSpecular){
            toggleAmbient = false;
            std::cout << "ambient turned off\n";}
        else if (!toggleAmbient && toggleDiffuse && toggleSpecular){
            toggleDiffuse = false;
            std::cout << "diffuse turned off as well\n";}
        else if (!toggleAmbient && !toggleDiffuse && toggleSpecular){
            toggleSpecular = false;
            std::cout << "specular turned off as well\n";}
        else{
            toggleAmbient = toggleDiffuse = toggleSpecular = true;
            std::cout << "everything turned back on\n";}
        oKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_O) == GLFW_RELEASE)
        oKeyPressed = false;

    // Toggle light mode: fixed or object-bound
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS && !lKeyPressed) {
        lightAttachedToObject = !lightAttachedToObject;
        std::cout << "lightAttachedToObject: " << (lightAttachedToObject ? "true" : "false") << "\n";
        lKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_RELEASE)
        lKeyPressed = false;

    // Toggle material type: plastic or metallic
    if (glfwGetKey(window, GLFW_KEY_M) == GLFW_PRESS && !mKeyPressed) {
        useMetallic = !useMetallic;
        std::cout << "useMetallic: " << (useMetallic ? "true" : "false") << "\n";
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
        if(renderMode==WIREFRAME){
            renderMode=SHADED;
        }
        else if (renderMode==SHADED){
            renderMode=TEXTURED;
        }
        else{
            renderMode= WIREFRAME;
        }
        tKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_T) == GLFW_RELEASE)
        tKeyPressed = false;

    // Toggle between different texture images
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !iKeyPressed) {
        currentTextureIndex = (currentTextureIndex + 1) % 2;
        iKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_I) == GLFW_RELEASE)
        iKeyPressed = false;


    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS && !rKeyPressed) {
        posX = -0.9f; 
        posY = 0.9f; 
        velocityY = 0.0f;
        rKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_RELEASE) {
        rKeyPressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !pKeyPressed) {
        if(isPaused){
            velocityX = velocityXsaved;
            velocityY = velocityYsaved;
            gravity= gravitySaved;
            isPaused= false;
        }
        else{
            velocityXsaved = velocityX;
            velocityYsaved = velocityY;
            gravitySaved = gravity;
            velocityX =0;
            velocityY =0;
            gravity =0;
            isPaused = true;
        }
        pKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_RELEASE) {
        pKeyPressed = false;
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_PRESS && !hKeyPressed) {
        std::cout << "Controls:\n";
        std::cout << "Press S to toggle shading mode (Gouraud/Phong).\n";
        std::cout << "Press O to toggle lighting components (ambient, diffuse, specular).\n";
        std::cout << "Press L to toggle light position (fixed/object-bound).\n";
        std::cout << "Press M to toggle material (plastic/metallic).\n";
        std::cout << "Press Z to zoom in, W to zoom out.\n";
        std::cout << "Press T to toggle rendering mode (shaded/wireframe/textured).\n";
        std::cout << "Press I to switch texture image.\n";
        std::cout << "Press R to reset ball position.\n";
        std::cout << "Press P to pause/unpause.\n";
        std::cout << "Press Q to quit.\n";

        hKeyPressed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_H) == GLFW_RELEASE) {
        hKeyPressed = false; 
    }
}