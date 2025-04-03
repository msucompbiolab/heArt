// GMSH Geo script to create thick LV septum wall

SetFactory("OpenCASCADE");


h = 2.0; //Meshsize
LVirad = 2.0; // LV internal radius
LVthick = 1.13; // LV thickness
Spthick = 2.0; // Septum thickness
LVolength = 8.3; // LV long axis length
RV_offset = 2.0; // RV offset from LV center
RVolength = 7.0; // RV long axis length
RVthick = LVthick/3; // RV thickness
RVirad = 3.2; // RV internal radius

LVspirad = LVirad - (Spthick - LVthick);
LVorad = LVirad + LVthick;
LVilength = LVolength - 0.5*LVthick;
RVi2rad = RVirad;


RVorad = RVirad + RVthick;
RVilength = RVolength - RVthick;
RVo2rad = RVi2rad + RVthick;



// Create LV geometry
Box(1) = {0.0, -LVorad, -LVorad, LVolength, 2*LVorad, 2*LVorad};

Sphere(2) = {0.0, 0.0, 0.0, 1.0};
Dilate {{0.0, 0.0 ,0.0}, {LVolength, LVorad, LVorad}}{
	Volume{2};
}

BooleanDifference(3) = {Volume{2}; Delete;}{Volume{1};};

Sphere(4) = {0.0, 0.0, 0.0, 1.0};
Dilate {{0.0, 0.0 ,0.0}, {LVilength, LVspirad , LVirad}}{
	Volume{4};
}

BooleanDifference(5) = {Volume{4}; Delete;}{Volume{1}; Delete;};

Sphere(400) = {0.0, 0.0, 0.0, 1.0};
Dilate {{0.0, 0.0 ,0.0}, {LVilength, LVirad, LVirad}}{
	Volume{400};
}
Box(100) = {0.0, -LVorad, -LVorad, LVolength, 2*LVorad, 2*LVorad};
BooleanDifference(500) = {Volume{400}; Delete;}{Volume{100}; Delete;};

Box(101) = {-LVolength, -LVorad, -LVorad, LVolength, LVorad, 2*LVorad};
BooleanIntersection(501) = {Volume{500}; Delete;}{Volume{101}; Delete;};

BooleanUnion(502) = {Volume{501}; Delete;}{Volume{5}; Delete;};


BooleanDifference(6) = {Volume{3}; Delete;}{Volume{502}; Delete;};

Box(100) = {0.0, -RVorad+RV_offset, -RVo2rad, RVolength, 2*RVorad, 2*RVo2rad};
Sphere(7) = {0.0, RV_offset, 0.0, 1};
Dilate {{0.0, RV_offset ,0.0}, {RVolength, RVorad, RVo2rad}}{
	Volume{7};
}
BooleanDifference(8) = {Volume{7}; Delete;}{Volume{100}; Delete;};

Box(101) = {0.0, -RVirad+RV_offset, -RVi2rad, RVilength, 2*RVirad, 2*RVi2rad};
Sphere(9) = {0.0, RV_offset, 0.0, 1};
Dilate {{0.0, RV_offset ,0.0}, {RVilength, RVirad, RVi2rad}}{
	Volume{9};
}
BooleanDifference(10) = {Volume{9}; Delete;}{Volume{101}; Delete;};
BooleanDifference(11) = {Volume{8}; Delete;}{Volume{10}; Delete;};

Sphere(13) = {0.0, 0.0, 0.0, 1.0};
Dilate {{0.0, 0.0 ,0.0}, {LVolength-0.3, LVorad-0.3, LVorad-0.3}}{
	Volume{13};
}

BooleanDifference(14) = {Volume{11}; Delete;}{Volume{13}; Delete;};


BooleanUnion(15) = {Volume{6}; Delete;}{Volume{14}; Delete;};

// field 1 computes the distance to your boundary
Field[1] = Attractor;
Field[1].EdgesList = {4,8,9,10,11,13,14,15,16,20,21,22,25,26};

// field 2 processes this distance to create a size field from 0.01 to 0.1
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = h/3;
Field[2].LcMax = h;
Field[2].DistMin = 0.1;
Field[2].DistMax = 0.5;
Background Field = 2;

//Physical Surface(1) = {5}; // LV Endo
//Physical Surface(2) = {1,3}; // Epi
//Physical Surface(3) = {6,4}; // RV Endo
//Physical Surface(4) = {2}; // Base
Physical Volume(1) = {15}; // Myocardium
