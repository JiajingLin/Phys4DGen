import openai
import base64
import json
from my_api_key import OPENAI_API_KEY

client = openai.OpenAI(
    api_key=OPENAI_API_KEY,
)

PRED_CAND_MATS_DENSITY_SYS_MSG_GPT4V_MULTI_IMAGES = """
I will provide you with multiple images for analysis. The first image presents a complete object, which we will label as the complete image. The other images describe parts of the complete object depicted in the first image.  

First, describe what the object in the complete image is. And which type of simulation (jelly, metal, sand, snow, plasticine) is most suitable to execute in the overall scene?

Then, sequentially and iteratively analyze each partial image. Each partial image must answer the following questions in order:  

First Question: Identify what the partial object in the partial image is. 

Second Question: Based on the first question, please first infer what material the surface of that partial object is most likely composed of; then, please infer what material the interior of that partial object is most likely composed of.

Finally Question: Based on the above information, please infer the most likely physical properties of the surface material and interior material of that partial object for the physical simulation. These properties include Density (kg/m³), Young’s modulus (Pa), and Poisson’s ratio. Provide a single value for each property. I will directly use these physical properties for simulation, so I need a reference value.

Format Requirement:

You must provide your answer in the following JSON format, as it will be parsed by a code script later. Your answer must look like:
{
		"complete object caption": description,
        "simulation type": selection (jelly、metal、sand、snow、plasticine)
		"partial object 1" :
		{
            "caption": description, (keep as concise as possible)
            "surface materials":
            {
	            "material name": description, (predict just one material and keep as concise as possible)
	            "physical property": {
	                "Density": simualtion reference value, 
	                "Young's modulus": simualtion reference value, (use scientific notation, e.g. 1.23e+05)
	                "Poisson's ratio": simualtion reference value
                }
            },
            "interior materials":
            {
                "material name": description, (predict just one material and keep as concise as possible),
                "physical property": {
                    "Density": simualtion reference value, 
                    "Young's modulus": simualtion reference value, (use scientific notation, e.g. 1.23e+05), 
                    "Poisson's ratio": simualtion reference value 
                }
            }
		},
		"partial object 2" : ......
}
You must be answer followed above requirement and do not include any other text and unnecessary words in your answer, as it will be parsed by a code script later.
"""

def parse_material_list(matlist, max_n=5):
    elems = matlist.split(';')
    if len(elems) > max_n:
        print('too many materials %s' % matlist)
        return None
    
    mat_names = []
    mat_vals = []

    for elem in elems:
        elem_parts = elem.strip().split(':')
        if len(elem_parts) != 2:
            print('bad format %s' % matlist)
            return None
        mat_name, values = elem_parts

        if not mat_name.startswith('(') or mat_name[1].isnumeric() or mat_name.startswith('(material 1'): 
            print('bad format %s' % matlist)
            return None

        mat_name = mat_name[1:]
        mat_names.append(mat_name.lower())

        values = values.strip().split(' ')[0]
        values = values.replace(",", "")
        if values[-1] == ')':
            values = values[:-1]

        splitted = values.split('-')
        try:
            float(splitted[0])
        except ValueError:
            print('value cannot be converted to float %s' % matlist)
            return None

        if len(splitted) == 2:
            mat_vals.append([float(splitted[0]), float(splitted[1])]) 
        elif len(splitted) == 1:
            mat_vals.append([float(splitted[0]), float(splitted[0])])
        else:
            print('bad format %s' % matlist)
            return None
        
    return mat_names, mat_vals


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def gpt4v_multi_images_predict(complete_image_path, partial_image_paths, model_name='gpt-4-vision-preview', seed=100):
    sys_msg = PRED_CAND_MATS_DENSITY_SYS_MSG_GPT4V_MULTI_IMAGES
    base64_complete_image = encode_image(complete_image_path)
    base64_partial_images = [encode_image(image_path) for image_path in partial_image_paths]
    messages_content = []
    messages_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{base64_complete_image}"
        }
    })
    for base64_partial_image in base64_partial_images:
        messages_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_partial_image}"
            }
        })
    response = client.chat.completions.create(
        model=model_name,
        messages=[
        {
            "role": "system",
            "content": sys_msg
        },
        {
            "role": "user",
            "content": messages_content
        }
        ],
        seed=seed,
        temperature=0.8,
    )
    return response.choices[0].message.content

def parse_json_response(json_string):
    try:
        print(json_string)
        if json_string.startswith('```json') and json_string.endswith('```'):
            json_string = json_string.split('```json')[1].split('```')[0]

        response_dict = json.loads(json_string)
        return response_dict
    except json.JSONDecodeError:
        print("error")
        return None

