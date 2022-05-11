# The objective is to display annotated LeontisWesthof base pairs (e.g. from RNAView and Cerny et al.'s program) and NtCs
# Also, to assist manual annotation of same.
# followed: https://github.com/nglviewer/nglview#usage
# pip3 install nglview
#
#import py3Dmol
#view = py3Dmol.view(query='pdb:1ubq') 
#view.setStyle({'cartoon':{'color':'spectrum'}}) 
#view 
# to get nglview to work, had to install anaconda on macbook pro. then:
# conda --version
# conda config --add channels conda-forge
# conda install nglview -c bioconda
# jupyter-nbextension enable nglview --py --sys-prefix



import nglview
view = nglview.show_pdbid("3pqr")  # load "3pqr" from RCSB PDB and display viewer widget
view
print ("hello")
view._display_image()

import pandas
# column Y is 25th letter in alphabet. that column has header "nearest_class_rotrans" Other headers are "nearest_class_proscoH" and "nearest_class_prosco"
# chains are in "chain1" and "chain2", residues are in "res1" and "res2". insertino codes are "ins1" and "ins2"
#pdbId = "1gid"
#pdbId = "3d2g"
#pdbId = "3rg5"
pdbId = "6ol3"


basePairingDataFrame = pandas.read_csv("pairing/"+pdbId+"_zcut2.5_assignment.csv")
print (basePairingDataFrame.columns)
basePairingDataFrame = basePairingDataFrame.reset_index()  # make sure indexes pair with number of rows
#for index, row in basePairingDataFrame.iterrows():
    #print(row['pdbid'], row['nr1'],row['nr2'])

view = nglview.NGLWidget()
view.add_pdbid(pdbId)

view.clear_representations()
view.add_representation('licorice', selection='not backbone', color='element', radius=.2, bondSpacing = 400)
view.add_representation('cartoon', selection='backbone', color='element')

atomPair = [ [ "253.P", "220.P" ], [ "220.P", "253.P" ] ];
#atomPair = [  [ "220.P", "253.P" ] ];

#view.add_representation( "distance", { atomPair: atomPair } );
#print (view.add_representation('licorice', selection='(253 OR 220) and #P', color='element', radius='3.2', bondSpacing = 400)["atom1"]["x"])
for index, row in basePairingDataFrame.iterrows():
    #print(row['pdbid'], row['nr1'],row['nr2'])
    atom1String = str(row['nr1']) + "C1'"
    atom2String = str(row['nr2']) + "C1'"
    saengerClass = str(row['nearest_class_rotrans']).split("_")[0] # split e.g. 20_A_U_c2_2A on "_" and take the zeroth element, in this example 20.
    # Saenger classes are listed here http://ndbserver.rutgers.edu/ndbmodule/legends/saenger.html
    #print (atom1String, atom2String,saengerClass)
    #if (saengerClass == 19) :
    if ((saengerClass == "19") or (saengerClass == "20") or (saengerClass == "8")): # WatsonCrick / WatsonCrick / Cis. 19:  purine-pyrimidine, 20 = pyrimidine-purine, 8 = purine-purine.
        view.add_distance(atom_pair=[[atom1String, atom2String]], label_color="black", labelType = "text", labelText = "my text", color = "blue")
    # for some reaason this does not work:
    #if ((saengerClass == "1") or (saengerClass == "3") or (saengerClass == "12") or (saengerClass == "14")): # WatsonCrick / WatsonCrick / Trans, offset. 
    #    view.add_distance(atom_pair=[[atom1String, atom2String]], label_color="black", labelType = "text", labelText = "my text", color = "orange")

#view.add_representation('spacefill', selection='61 OR 253', color='blue', radius='1.2', bondSpacing = 40)
#view.get_coordinates() # takes an index
#view.shape.add_cylinder( [ 0, 2, 7 ], [ 0, 0, 39 ], [ 1, 1, 0 ], 0.5)
view
