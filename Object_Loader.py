import numpy as np
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
class Object_Loader:
    def __init__(self, path,x, y, z, factor = 1.0):
        self.path = path
        self.factor = factor
        self.x = x
        self.y = y
        self.z = z
        self.vert_coords = []
        self.tex_coords = []
        self.norm_coords = []
        self.indices = []
        self.buffer = []
        self.all_indices = []
        self.load_model(self,file = self.path)
        self.translate(self.x,self.y,self.z)
        
    @staticmethod # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(self,indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0: # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                self.buffer.extend(vertices[start:end])
            elif i % 3 == 1: # sort the texture coordinates
                start = ind * 2
                end = start + 2
                self.buffer.extend(textures[start:end])
            elif i % 3 == 2: # sort the normal vectors
                start = ind * 3
                end = start + 3
                self.buffer.extend(normals[start:end])


    @staticmethod # TODO unsorted vertex buffer for use with glDrawElements function
    def create_unsorted_vertex_buffer(indices_data, vertices, textures, normals):
        num_verts = len(vertices) // 3

        for i1 in range(num_verts):
            start = i1 * 3
            end = start + 3
            Object_Loader.buffer.extend(vertices[start:end])

            for i2, data in enumerate(indices_data):
                if i2 % 3 == 0 and data == i1:
                    start = indices_data[i2 + 1] * 2
                    end = start + 2
                    Object_Loader.buffer.extend(textures[start:end])

                    start = indices_data[i2 + 2] * 3
                    end = start + 3
                    Object_Loader.buffer.extend(normals[start:end])

                    break



        
    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d)-1)

    @staticmethod
    def load_model(self,file, sorted=True):
       
        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == 'v':
                    Object_Loader.search_data(values, self.vert_coords, 'v', 'float')
                elif values[0] == 'vt':
                    Object_Loader.search_data(values, self.tex_coords, 'vt', 'float')
                elif values[0] == 'vn':
                    Object_Loader.search_data(values, self.norm_coords, 'vn', 'float')
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')
                        Object_Loader.search_data(val, self.all_indices, 'f', 'int')
                        self.indices.append(int(val[0])-1)

                line = f.readline()

        if sorted:
            # use with glDrawArrays
            Object_Loader.create_sorted_vertex_buffer(self,self.all_indices, self.vert_coords, self.tex_coords, self.norm_coords)
        else:
            # use with glDrawElements
            Object_Loader.create_unsorted_vertex_buffer(self.all_indices, self.vert_coords, self.tex_coords, self.norm_coords)

        # self.show_buffer_data(Object_Loader.buffer)

        buffer = self.buffer.copy() # create a local copy of the buffer list, otherwise it will overwrite the static field buffer
        # Object_Loader.buffer = [] # after copy, make sure to set it back to an empty list

        # return np.array(self.indices, dtype='uint32'), np.array(buffer, dtype='float32')
        self.scale(self.factor)
    
    def translate(self, x, y, z):
        for i in range(0, len(self.vert_coords), 3):
            self.vert_coords[i] += x
            self.vert_coords[i + 1] += y
            self.vert_coords[i + 2] += z

    def scale(self, factor):
        for i in range(0, len(self.vert_coords), 3):
            self.vert_coords[i] *= factor
            self.vert_coords[i + 1] *= factor
            self.vert_coords[i + 2] *= factor

    def rotation(self, angle, axis):
        if axis == 'x':
            for i in range(0, len(self.vert_coords), 3):
                self.vert_coords[i + 1] = self.vert_coords[i + 1] * np.cos(angle) - self.vert_coords[i + 2] * np.sin(angle)
                self.vert_coords[i + 2] = self.vert_coords[i + 1] * np.sin(angle) + self.vert_coords[i + 2] * np.cos(angle)
                
        elif axis == 'y':
            for i in range(0, len(self.vert_coords), 3):
                self.vert_coords[i] = self.vert_coords[i] * np.cos(angle) + self.vert_coords[i + 2] * np.sin(angle)
                self.vert_coords[i + 2] = -self.vert_coords[i] * np.sin(angle) + self.vert_coords[i + 2] * np.cos(angle)
                
        elif axis == 'z':
            for i in range(0, len(self.vert_coords), 3):
                self.vert_coords[i] = self.vert_coords[i] * np.cos(angle) - self.vert_coords[i + 1] * np.sin(angle)
                self.vert_coords[i + 1] = self.vert_coords[i] * np.sin(angle) + self.vert_coords[i + 1] * np.cos(angle)
                

    def draw(self):
        # Load data into OpenGL buffers
        self.vert_coords = np.array(self.vert_coords, dtype='float32')
        self.tex_coords = np.array(self.tex_coords, dtype='float32')
        self.norm_coords = np.array(self.norm_coords, dtype='float32')
        self.indices = np.array(self.indices, dtype='uint32')
        vao = glGenVertexArrays(1)
        glBindVertexArray(vao)

        vbo = glGenBuffers(3)
        glBindBuffer(GL_ARRAY_BUFFER, vbo[0])
        glBufferData(GL_ARRAY_BUFFER, self.vert_coords.nbytes, self.vert_coords, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(0)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[1])
        glBufferData(GL_ARRAY_BUFFER, self.tex_coords.nbytes, self.tex_coords, GL_STATIC_DRAW)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(1)

        glBindBuffer(GL_ARRAY_BUFFER, vbo[2])
        glBufferData(GL_ARRAY_BUFFER, self.norm_coords.nbytes, self.norm_coords, GL_STATIC_DRAW)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, None)
        glEnableVertexAttribArray(2)

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, glGenBuffers(1))
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        # Create shader program
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
        shader_program = compileProgram(compileShader(vertex_shader, GL_VERTEX_SHADER),
        compileShader(fragment_shader, GL_FRAGMENT_SHADER))
        glUseProgram(shader_program)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(self.indices), GL_UNSIGNED_INT, None)