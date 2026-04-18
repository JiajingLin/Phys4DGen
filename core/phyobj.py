import json

class PhyObj:
    class PhyMaterial:
        def __init__(self, info):
            self.material_name = info['material name']
            self.density = info['physical property']['Density']
            self.youngs_modulus = info['physical property']['Young\'s modulus']
            self.poissons_ratio = info['physical property']['Poisson\'s ratio']
            self.thickness = info['thickness'] if 'thickness' in info else None
    
    class PhyPartObj:
        def __init__(self, info):
            self.caption = info['caption']
            # self.isempty = info['isempty']
            self.surface_materials = []
            self.internal_materials = []
            for key in info.keys():
                if key.startswith('surface materials'):
                    self.surface_materials.append(PhyObj.PhyMaterial(info[key]))
                elif key.startswith('interior materials'):
                    self.internal_materials.append(PhyObj.PhyMaterial(info[key]))

    def __init__(self, info_file):
        
        with open(info_file, 'r') as f:
            infos = json.load(f)

        self.caption = infos['complete object caption']
        if 'simulation type' in infos:
            self.simulation_type = infos['simulation type']  
        else:
            self.simulation_type = None
            
        self.part_objs = []
        for part in infos.keys():
            if part.startswith('partial object'):
                self.part_objs.append(PhyObj.PhyPartObj(infos[part]))
                
        
    def get_partobjs_caption(self):
        return [part.caption for part in self.part_objs]
    
    
    def get_partobjs_surface_property(self, property_name):
        if property_name == 'Density':
            return [part.surface_materials[0].density for part in self.part_objs]
        elif property_name == "Young's modulus":
            return [part.surface_materials[0].youngs_modulus for part in self.part_objs]
        elif property_name == "Poisson's ratio":
            return [part.surface_materials[0].poissons_ratio for part in self.part_objs]
    
    
    def get_partobjs_internal_property(self, property_name):
        
        if property_name == 'Density':
            return [part.internal_materials[0].density for part in self.part_objs]
        elif property_name == "Young's modulus":
            return [part.internal_materials[0].youngs_modulus for part in self.part_objs]
        elif property_name == "Poisson's ratio":
            return [part.internal_materials[0].poissons_ratio for part in self.part_objs]
