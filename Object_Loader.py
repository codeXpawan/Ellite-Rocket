import numpy as np
from OpenGL.GL import *
import pywavefront
from OpenGL.GL.shaders import compileProgram, compileShader
class Object_Loader:
    def __init__(self, path,x, y, z, factor = 1.0):
        self.path = path
        self.factor = factor
        self.x = x
        self._wavefront = None
        self.vaos = None
        self._vbos = None
        self.materials = []
        self.textures = None
        self.use_texture = False
        self.lengths = []
        self.y = y
        self.z = z
        self.vert_coords = []
        self.tex_coords = []
        self.norm_coords = []
        self.indices = []
        self.buffer = []
        self.all_indices = []
        # self.load_model(self,file = self.path)
        
        # self.translate(self.x,self.y,self.z)
        # self.vertices = self.loadMesh(self.path)
        
        # self.vertex_count = len(self.vertices)
        # self.vertices = np.array(self.vertices, dtype=np.float32)
        self._load_obj(factor)
        
    
    def _load_obj(self, scale_factor: float = 1.0) -> None:
        """Loads wavefront obj and materials. Stores vertex data into VAOs and VBOs."""
        self._wavefront = pywavefront.Wavefront(self.path, collect_faces=True, create_materials=True)
        
        # Generate buffers
        materials_count = len(self._wavefront.materials)
        self.vaos = glGenVertexArrays(materials_count)  #to store pointer to different VBOs and switch whenever necessary
        self._vbos = glGenBuffers(materials_count)      #generate buffer for each material
        self.textures = glGenTextures(materials_count)

        if materials_count == 1:
            # glGen* will return an int instead of an np.array if argument is 1.
            self.vaos = np.array([self.vaos], dtype=np.uint32)  #for loading to GPU
            self._vbos = np.array([self._vbos], dtype=np.uint32)
            self.textures = np.array([self.textures], dtype=np.uint32)

        # For each material fill buffers and load a texture
        ind = 0
        for material in self._wavefront.materials.values():
            vertex_size = material.vertex_size
            scene_vertices = np.array(material.vertices, dtype=np.float32) * scale_factor
            print(scene_vertices)
            # Store length and materials for drawing
            self.lengths.append(len(scene_vertices))
            self.materials.append(material)
            # Load texture by path 
            if material.texture is not None:
                self._load_texture(material.texture.path, self.textures[ind])
                self.use_texture = True
            
            # Bind VAO
            glBindVertexArray(self.vaos[ind])
            # Fill VBO
            glBindBuffer(GL_ARRAY_BUFFER, self._vbos[ind])
            glBufferData(GL_ARRAY_BUFFER, scene_vertices.nbytes, scene_vertices, GL_STATIC_DRAW) #Store vertices in buffer

            # Set attribute buffers
            attr_format = {
                "T2F": (1, 2),  # Tex coords (2 floats): ind=1
                "C3F": (2, 3),  # Color (3 floats): ind=2
                "N3F": (3, 3),  # Normal (3 floats): ind=3
                "V3F": (0, 3),  # Position (3 floats): ind=0
            }
    
            cur_off = 0  # current start offset
            for attr in material.vertex_format.split("_"):
                if attr not in attr_format:
                    raise Exception("Unknown format")

                # Apply
                attr_ind, attr_size = attr_format[attr]
                glEnableVertexAttribArray(attr_ind)
                glVertexAttribPointer(attr_ind, attr_size, GL_FLOAT, GL_FALSE, scene_vertices.itemsize * vertex_size, #amount of data between each data
                                      ctypes.c_void_p(cur_off))  #pointer to where vertices begin in array
                cur_off += attr_size * 4

            # Unbind (Technically not necessary but used as a precaution)
            glBindVertexArray(0)
            ind += 1

    @staticmethod
    def _load_texture(path: str, texture: int) -> None:
        """
        Loads texture into buffer by given path and tex buffer ID.

        :param path: Texture path.
        :param texture: Texture buffer ID.
        """
        # For use with GLFW
        glBindTexture(GL_TEXTURE_2D, texture)
        # Set the texture wrapping parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # Set texture filtering parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Load image
        # image = Image.open(path)
        # image = image.transpose(Image.FLIP_TOP_BOTTOM)
        # img_data = image.convert("RGBA").tobytes()
        # glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

    def drew(self) -> None:
        """Draws loaded object onto GL buffer with selected shader."""
        # shader.use_program()  # Not really sure if that's how you should do it
        vertex_shader = '''
        #version 330

        layout (location = 0) in vec3 position;
        layout (location = 1) in vec2 texcoord;
        layout (location = 2) in vec3 normal;

        out vec2 frag_texcoord;
        out vec3 frag_normal;

        void main()
        {
            gl_Position = vec4(position, 1.0);
            frag_texcoord = texcoord;
            frag_normal = normal;
        }
        '''
            
        fragment_shader = '''
        #version 330

        in vec2 frag_texcoord;
        in vec3 frag_normal;

        out vec4 out_color;

        void main()
        {
        vec3 light_dir = normalize(vec3(1.0, 1.0, 1.0));
        vec3 normal = normalize(frag_normal);
        float diffuse = max(dot(normal, light_dir), 0.0);
        out_color = vec4(vec3(1.0) * diffuse, 1.0);
        }
        '''
        for vao, tex, length, mat in zip(self.vaos, self.textures, self.lengths, self.materials):
            glBindVertexArray(vao) #Bind VAO
            glBindTexture(GL_TEXTURE_2D, tex)
            # if model is not None:
            #     shader.set_model(model)
            # else:
            #     shader.set_model(Arithmetic.multiply(self._scale_matrix, self.model))
            # shader.set_v3("material.ambient", mat.ambient)
            # shader.set_v3("material.diffuse", mat.diffuse)
            # shader.set_v3("material.specular", mat.specular)
            # shader.set_float("material.shininess", mat.shininess)
            
            # shader_program = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER),
            # compileShader(fragment_shader, GL_FRAGMENT_SHADER))
            # glUseProgram(shader_program)
            glDrawArrays(GL_TRIANGLES, 0, length)

