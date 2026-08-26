% Export MATLAB VFEM3D summaries for Julia cross-checking.
clear;

% Point VFEM_LIB_ROOT at your checkout of the MATLAB VFEM_LIB/VFEM3D tree.
root = getenv('VFEM_LIB_ROOT');
if isempty(root), error('set VFEM_LIB_ROOT to the MATLAB VFEM3D directory'); end
addpath(genpath(root));
cd(root);
my_intlab_mode_config;

% Point VFEM3D_ROOT at your checkout of this repository.
vfem3d_root = getenv('VFEM3D_ROOT');
if isempty(vfem3d_root), vfem3d_root = pwd; end
out_file = fullfile(vfem3d_root, 'examples', 'reference', 'vfem3d_matlab_debug.txt');

nodes = [ 0.0   0.0   0.0;
          0.0   0.0   1.0;
          0.5   0.5   0.5;
         -0.5   0.5   0.5;
          0.0   0.25  0.5 ];
tets = [2 3 4 5;
        1 3 4 5;
        1 2 4 5;
        1 2 3 5];

mesh.NodeList = nodes;
mesh.ElementList = tets;
mesh.FacetList = get_FacetList(mesh.ElementList);
[mesh.EdgeList, mesh.NumEdge] = mesh_get_EdgeList(mesh, 100);
mesh.NumNode = size(nodes, 1);
mesh.NumElt = size(tets, 1);
mesh.NumF = size(mesh.FacetList, 1);
[mesh.Facet2Element, mesh.Element2Facet] = mesh_get_Facet2Element(mesh);

fid = fopen(out_file, 'w');
fprintf(fid, 'NumNode=%d\n', mesh.NumNode);
fprintf(fid, 'NumElt=%d\n', mesh.NumElt);
fprintf(fid, 'NumF=%d\n', mesh.NumF);
fprintf(fid, 'NumEdge=%d\n', mesh.NumEdge);

% ---- CG2 Lagrange stiffness/mass and eigenvalue.
p = 2;
[L2G, DOF_BD, DimCG] = FEM_space_register_cg_dof_v2(mesh, p);
interior = setdiff(1:DimCG, DOF_BD);
DegK = get_DOF(3, p);
Aref = getInnerProdMatrix_Reference(p, p);
Gref = getInnerProdMatrix_Reference(p-1, p-1);
nnz_est = mesh.NumElt * DegK^2;
iK = zeros(nnz_est, 1); jK = zeros(nnz_est, 1); vK = zeros(nnz_est, 1);
iM = zeros(nnz_est, 1); jM = zeros(nnz_est, 1); vM = zeros(nnz_est, 1);
ptr = 0;
for e = 1:mesh.NumElt
    LocalNodes = mesh.NodeList(mesh.ElementList(e,:), :);
    vol = get_volume(LocalNodes);
    MatGrad = get_GradMat(p, LocalNodes);
    Kloc = zeros(DegK, DegK);
    for d = 1:3
        Kloc = Kloc + double(MatGrad(:,:,d))' * double(Gref) * double(MatGrad(:,:,d));
    end
    Kloc = Kloc * vol;
    Mloc = double(Aref) * vol;
    dofs = L2G(e, :);
    [jloc, iloc] = meshgrid(1:DegK, 1:DegK);
    nblock = DegK^2;
    iK(ptr+(1:nblock)) = dofs(iloc(:)); jK(ptr+(1:nblock)) = dofs(jloc(:)); vK(ptr+(1:nblock)) = Kloc(:);
    iM(ptr+(1:nblock)) = dofs(iloc(:)); jM(ptr+(1:nblock)) = dofs(jloc(:)); vM(ptr+(1:nblock)) = Mloc(:);
    ptr = ptr + nblock;
end
K = sparse(iK(1:ptr), jK(1:ptr), vK(1:ptr), DimCG, DimCG);
M = sparse(iM(1:ptr), jM(1:ptr), vM(1:ptr), DimCG, DimCG);
lam = sort(real(eig(full(K(interior,interior)), full(M(interior,interior)))));
fprintf(fid, 'CG2_DimCG=%d\n', DimCG);
fprintf(fid, 'CG2_NumBD=%d\n', length(DOF_BD));
fprintf(fid, 'CG2_NumInt=%d\n', length(interior));
fprintf(fid, 'CG2_A_trace=%.17g\n', full(trace(K)));
fprintf(fid, 'CG2_A_sum=%.17g\n', full(sum(K(:))));
fprintf(fid, 'CG2_A_frob=%.17g\n', norm(full(K), 'fro'));
fprintf(fid, 'CG2_M_trace=%.17g\n', full(trace(M)));
fprintf(fid, 'CG2_M_sum=%.17g\n', full(sum(M(:))));
fprintf(fid, 'CG2_M_frob=%.17g\n', norm(full(M), 'fro'));
fprintf(fid, 'CG2_lambda1=%.17g\n', lam(1));

% ---- RT1 mixed matrices.
rt = build_scalar_rt_matrices(mesh, 1);
fprintf(fid, 'RT1_DimRT=%d\n', rt.DimRT);
fprintf(fid, 'RT1_DimDG=%d\n', rt.DimDG);
fprintf(fid, 'RT1_DegK=%d\n', rt.DegK);
fprintf(fid, 'RT1_A_trace=%.17g\n', full(trace(rt.A_rt)));
fprintf(fid, 'RT1_A_sum=%.17g\n', full(sum(rt.A_rt(:))));
fprintf(fid, 'RT1_A_frob=%.17g\n', norm(full(rt.A_rt), 'fro'));
fprintf(fid, 'RT1_B_sum=%.17g\n', full(sum(rt.B_rt(:))));
fprintf(fid, 'RT1_B_frob=%.17g\n', norm(full(rt.B_rt), 'fro'));
fprintf(fid, 'RT1_Mdg_trace=%.17g\n', full(trace(rt.M_dg)));
fprintf(fid, 'RT1_Mdg_sum=%.17g\n', full(sum(rt.M_dg(:))));
fprintf(fid, 'RT1_Mdg_frob=%.17g\n', norm(full(rt.M_dg), 'fro'));

fclose(fid);
fprintf('Wrote %s\n', out_file);
