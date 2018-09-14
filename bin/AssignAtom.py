
import numpy as np
import os.path
import re

fname = '../../../PERM_Struct/XSMD/cytc_wb_Cl.psf'
with open(fname) as f:
    PSF = f.readlines()

PSF = [x.strip() for x in PSF]

fname = 'param.cu'
with open(fname) as f:
    TYP = f.readlines()

TYP = [x.strip() for x in TYP]

#print(PSF)
get_bonds_from_now_on = 0
get_types_from_now_on = 0

for x in PSF:
    if (get_bonds_from_now_on == 1):
        #print(idx)
        temp_bonds = re.findall('\d+', x)
        temp_bonds = map(int, temp_bonds)
        while (idx < NBOND):
            bonds[idx][:] = temp_bonds[0:2]
            #print(bonds[idx][:])
            idx = idx + 1
            if (len(temp_bonds) > 2):
                temp_bonds[0:2] = []
            else:
                break
    if '!NBOND:' in x:
        print('NBONDS found.')
        print(x)
        print(re.search(r'\d+', x).group())
        NBOND = int(re.search(r'\d+', x).group())
        print('There are {:d} bonds.'.format(NBOND))
        get_bonds_from_now_on = 1
        idx = 0
        bonds = np.zeros((NBOND,2),dtype=int)

for x in TYP:
    if 'Ele' in x:
        print('Element list found.')
        #print(x)
        print(re.search(r'\d+', x).group())
        NATOM = int(re.search(r'\d+', x).group())
        print('There are {:d} atoms.'.format(NATOM))
        x = x.split('{')[-1]
        x = x.split('}')[0]
        x = x.split(',')
        x = map(int, x)
        Ele = x
        #print(Ele)


bonds = bonds.flatten()
HC = 0
HN = 0
HO = 0
HS = 0
num_ele = np.max(Ele)

for idx, atom in enumerate(Ele):
    if atom == 0:
#        print('Idx is {:d}'.format(idx))
        atom_H = bonds.tolist().index(idx+1)
#        print('atom_H index is {:d}'.format(atom_H))
        if (atom_H % 2 == 0):
            atom_X = atom_H + 1
        else:
            atom_X = atom_H - 1
#        print('atom_X index is {:d}'.format(atom_X))
        if (Ele[bonds[atom_X]-1] == 1):
            A = 'Carbon'
            HC = HC + 1
        elif (Ele[bonds[atom_X]-1] == 2):
            A = 'Nitrogen'
            HN = HN + 1
        elif (Ele[bonds[atom_X]-1] == 3):
            A = 'Oxygen'
            HO = HO + 1
        elif (Ele[bonds[atom_X]-1] == 4):
            A = 'Sulfur'
            HS = HS + 1

#        print('Corresponding heavy atom is {:d} ({:s})'.format(Ele[bonds[atom_X]-1],A))
        Ele[idx] = Ele[idx] + num_ele + Ele[bonds[atom_X]-1]

print(Ele)
#print('There are {:d} HCs'.format(HC))
#print('There are {:d} HNs'.format(HN))
#print('There are {:d} HOs'.format(HO))
#print('There are {:d} HSs'.format(HS))




