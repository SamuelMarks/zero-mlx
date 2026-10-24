# Compiler Operations Test Parity

This tracks which ml-switcheroo-compiler operations have a dedicated parity test against MLX in `tests/test_ops_parity_*.py`.

| Status | Operation |
|---|---|
| [x] | `Abs` |
| [x] | `AccumulateN` |
| [x] | `Acos` |
| [x] | `Acosh` |
| [x] | `ActivityRegularization` |
| [ ] | `AdamaxHyperparams` |
| [x] | `AdaptiveAvgPool2D` |
| [ ] | `AdaptiveAvgPool3D` |
| [x] | `AdaptiveMaxPool2D` |
| [ ] | `AdaptiveMaxPool3D` |
| [ ] | `AdaptiveMaxPool3D_Indices` |
| [x] | `Add` |
| [x] | `AddN` |
| [x] | `Adjoint` |
| [x] | `AdjustBrightness` |
| [x] | `AdjustContrast` |
| [x] | `AdjustHue` |
| [x] | `AdjustSaturation` |
| [x] | `AffineGenerator` |
| [x] | `AffineGrid` |
| [x] | `AffineTransform` |
| [x] | `All` |
| [x] | `Allclose` |
| [x] | `AlphaDropout` |
| [x] | `Angle` |
| [x] | `Any` |
| [ ] | `Append` |
| [ ] | `ApplyAdagrad` |
| [ ] | `ApplyAdam` |
| [ ] | `ApplyFtrl` |
| [ ] | `ApplyOverAxes` |
| [ ] | `ApplyRMSProp` |
| [x] | `Argmax` |
| [x] | `Argmin` |
| [x] | `Argpartition` |
| [ ] | `Argsort` |
| [x] | `Argwhere` |
| [ ] | `ArrayEquiv` |
| [ ] | `ArrayRepr` |
| [ ] | `ArrayStr` |
| [x] | `AsString` |
| [x] | `Asin` |
| [x] | `Asinh` |
| [x] | `Assert` |
| [x] | `Assign` |
| [x] | `AssignAdd` |
| [x] | `AssignSub` |
| [x] | `AssignVariable` |
| [x] | `AssociativeScan` |
| [x] | `Atan` |
| [x] | `Atan2` |
| [x] | `Atanh` |
| [ ] | `Atleast1d` |
| [ ] | `Atleast2d` |
| [ ] | `Atleast3d` |
| [x] | `AugMix` |
| [x] | `AutoContrast` |
| [ ] | `Average` |
| [x] | `Ball` |
| [x] | `BandPart` |
| [x] | `BandedTriangularSolve` |
| [x] | `BesselI0` |
| [x] | `BesselI0e` |
| [x] | `BesselI1` |
| [x] | `BesselI1e` |
| [x] | `BesselJ0` |
| [x] | `BesselJ1` |
| [x] | `BesselJn` |
| [x] | `BesselK0` |
| [x] | `BesselK0e` |
| [x] | `BesselK1` |
| [x] | `BesselK1e` |
| [x] | `BesselY0` |
| [x] | `BesselY1` |
| [x] | `Beta` |
| [ ] | `BetaCdf` |
| [ ] | `BetaPdf` |
| [x] | `Betainc` |
| [x] | `Bincount` |
| [ ] | `BinomCdf` |
| [ ] | `BinomPmf` |
| [x] | `Binomial` |
| [x] | `Bitcast` |
| [x] | `BitcastConvertType` |
| [x] | `Bits` |
| [x] | `BitwiseAnd` |
| [x] | `BitwiseCount` |
| [x] | `BitwiseNot` |
| [x] | `BitwiseOr` |
| [x] | `BitwiseXor` |
| [ ] | `Block` |
| [x] | `BlockMaskedMm` |
| [x] | `BooleanMask` |
| [x] | `BroadcastInDim` |
| [x] | `BroadcastTo` |
| [ ] | `C` |
| [x] | `CTCLoss` |
| [ ] | `CanCast` |
| [x] | `Cast` |
| [x] | `Cauchy` |
| [x] | `Cbrt` |
| [x] | `Ceil` |
| [ ] | `ChebyshevPolynomialT` |
| [ ] | `ChebyshevPolynomialU` |
| [x] | `Chisquare` |
| [x] | `Cholesky` |
| [ ] | `CholeskyEx` |
| [x] | `CholeskySolve` |
| [x] | `Choose` |
| [ ] | `Clip` |
| [x] | `Clone` |
| [x] | `Clz` |
| [ ] | `Collapse` |
| [x] | `ColumnStack` |
| [x] | `Compress` |
| [x] | `Concatenate` |
| [ ] | `Cond` |
| [x] | `ConfusionMatrix` |
| [x] | `Conj` |
| [x] | `ConvGeneralDilated` |
| [x] | `ConvGeneralDilatedLocal` |
| [x] | `ConvGeneralDilatedPatches` |
| [x] | `ConvTranspose` |
| [ ] | `ConvTransposeShapeTuple` |
| [x] | `ConvWithGeneralPadding` |
| [x] | `Convolve` |
| [ ] | `Convolve2d` |
| [x] | `Copysign` |
| [x] | `Corrcoef` |
| [x] | `Correlate` |
| [x] | `Cos` |
| [x] | `Cosh` |
| [x] | `CountNonzero` |
| [x] | `Cov` |
| [ ] | `CreateToken` |
| [x] | `Cross` |
| [x] | `Cumlogsumexp` |
| [x] | `Cummax` |
| [x] | `Cummin` |
| [x] | `Cumprod` |
| [x] | `Cumsum` |
| [x] | `CumulativeLogsumexp` |
| [x] | `CustomLinearSolve` |
| [x] | `CustomRoot` |
| [x] | `Dawsn` |
| [x] | `DebugInfs` |
| [x] | `DebugNans` |
| [ ] | `DecodeBase64` |
| [ ] | `DecodeCsv` |
| [ ] | `DecodeImage` |
| [x] | `Deg2Rad` |
| [ ] | `Degrees` |
| [x] | `Delete` |
| [ ] | `Descriptive` |
| [x] | `Det` |
| [ ] | `DeviceContext` |
| [ ] | `DeviceTransfer` |
| [x] | `Diag` |
| [x] | `DiagIndices` |
| [x] | `DiagIndicesFrom` |
| [x] | `Diagflat` |
| [x] | `Diagonal` |
| [x] | `Diff` |
| [x] | `Digamma` |
| [x] | `Digitize` |
| [x] | `Dirichlet` |
| [ ] | `Distributions` |
| [x] | `Divide` |
| [x] | `DivideNoNan` |
| [x] | `Divmod` |
| [x] | `Dot` |
| [x] | `DotGeneral` |
| [x] | `DoubleSidedMaxwell` |
| [x] | `Dropout` |
| [ ] | `Dropout1d` |
| [x] | `Dropout2d` |
| [x] | `Dropout3d` |
| [x] | `Dsplit` |
| [x] | `Dstack` |
| [x] | `DynamicIndexInDim` |
| [x] | `DynamicPartition` |
| [x] | `DynamicShape` |
| [x] | `DynamicSlice` |
| [x] | `DynamicSliceInDim` |
| [x] | `DynamicStitch` |
| [x] | `DynamicUpdateIndexInDim` |
| [x] | `DynamicUpdateSlice` |
| [x] | `DynamicUpdateSliceInDim` |
| [ ] | `Ediff1d` |
| [x] | `EditDistance` |
| [ ] | `Eig` |
| [x] | `Eigh` |
| [x] | `EighTridiagonal` |
| [ ] | `Eigvals` |
| [x] | `Eigvalsh` |
| [x] | `Einsum` |
| [ ] | `EinsumPath` |
| [x] | `ElasticTransform` |
| [ ] | `EncodeBase64` |
| [x] | `Equal` |
| [x] | `Equalization` |
| [x] | `Erf` |
| [x] | `Erfc` |
| [x] | `Erfcinv` |
| [x] | `Erfinv` |
| [x] | `Exp` |
| [x] | `Exp2` |
| [ ] | `ExpandDims` |
| [x] | `Expint` |
| [x] | `Expm1` |
| [x] | `Exponential` |
| [ ] | `Extract` |
| [x] | `ExtractBoundingBoxes` |
| [x] | `ExtractVolumePatches` |
| [x] | `F` |
| [ ] | `Fabs` |
| [x] | `Fft` |
| [ ] | `Fft2` |
| [ ] | `Fftconvolve` |
| [x] | `Fftfreq` |
| [ ] | `Fftn` |
| [x] | `Fftnd` |
| [x] | `Fftshift` |
| [ ] | `FillDiagonal` |
| [x] | `Fix` |
| [ ] | `Flatnonzero` |
| [ ] | `Flip` |
| [ ] | `Fliplr` |
| [ ] | `Flipud` |
| [x] | `FloatPower` |
| [x] | `Floor` |
| [x] | `FloorDivide` |
| [x] | `Fmax` |
| [x] | `Fmin` |
| [x] | `Fmod` |
| [x] | `Fold` |
| [ ] | `FractionalAvgPool` |
| [ ] | `FractionalMaxPool` |
| [x] | `FractionalMaxPool2D` |
| [ ] | `FractionalMaxPool3D` |
| [ ] | `FractionalMaxPool3D_Indices` |
| [x] | `FresnelCos` |
| [x] | `FresnelSin` |
| [x] | `Frexp` |
| [ ] | `FromDlpack` |
| [ ] | `Fromfile` |
| [ ] | `Fromfunction` |
| [ ] | `Fromiter` |
| [ ] | `Frompyfunc` |
| [ ] | `Fromstring` |
| [x] | `Gamma` |
| [ ] | `GammaCdf` |
| [ ] | `GammaPdf` |
| [x] | `Gather` |
| [x] | `GatherMm` |
| [x] | `GatherNd` |
| [x] | `Gcd` |
| [x] | `GeneralizedNormal` |
| [ ] | `Geometric` |
| [ ] | `Geomspace` |
| [x] | `GetItem` |
| [x] | `Greater` |
| [x] | `GreaterEqual` |
| [x] | `GroupMean` |
| [x] | `GroupNorm` |
| [x] | `GroupVariance` |
| [x] | `Gru` |
| [x] | `Gumbel` |
| [x] | `Hashing` |
| [x] | `Heaviside` |
| [ ] | `HermitePolynomialH` |
| [ ] | `HermitePolynomialHe` |
| [x] | `Hessenberg` |
| [x] | `Hfft` |
| [x] | `HouseholderProduct` |
| [x] | `Hsplit` |
| [x] | `Hstack` |
| [x] | `Hypot` |
| [x] | `Ifft` |
| [ ] | `Ifft2` |
| [ ] | `Ifftn` |
| [x] | `Ifftnd` |
| [x] | `Ifftshift` |
| [x] | `Igamma` |
| [x] | `IgammaGradA` |
| [x] | `Igammac` |
| [x] | `Ihfft` |
| [x] | `Imag` |
| [ ] | `IndexInDim` |
| [x] | `Inner` |
| [ ] | `Insert` |
| [x] | `IntegerLookup` |
| [x] | `Inv` |
| [ ] | `InvEx` |
| [x] | `Invert` |
| [x] | `InvertPermutation` |
| [ ] | `Iou` |
| [x] | `Irfft` |
| [ ] | `Irfft2` |
| [ ] | `Irfftn` |
| [x] | `Irfftnd` |
| [x] | `IsNonDecreasing` |
| [x] | `IsStrictlyIncreasing` |
| [x] | `Isclose` |
| [ ] | `Iscomplex` |
| [ ] | `Iscomplexobj` |
| [x] | `Isfinite` |
| [ ] | `Isin` |
| [x] | `Isinf` |
| [x] | `Isnan` |
| [ ] | `Isneginf` |
| [ ] | `Isposinf` |
| [ ] | `Isreal` |
| [ ] | `Isrealobj` |
| [ ] | `Issubdtype` |
| [x] | `Istft` |
| [x] | `Key` |
| [x] | `KeyData` |
| [x] | `KeyImpl` |
| [x] | `L2Normalize` |
| [ ] | `LaguerrePolynomialL` |
| [x] | `Laplace` |
| [x] | `Lbeta` |
| [x] | `Lcm` |
| [x] | `Ldexp` |
| [x] | `LeftShift` |
| [ ] | `LegendrePolynomialP` |
| [x] | `Less` |
| [x] | `LessEqual` |
| [ ] | `Lexsort` |
| [x] | `Lgamma` |
| [ ] | `LinearOperator` |
| [ ] | `LinearOperatorAdjoint` |
| [ ] | `LinearOperatorBlockDiag` |
| [ ] | `LinearOperatorBlockLowerTriangular` |
| [ ] | `LinearOperatorCirculant` |
| [ ] | `LinearOperatorCirculant2D` |
| [ ] | `LinearOperatorCirculant3D` |
| [ ] | `LinearOperatorComposition` |
| [ ] | `LinearOperatorDiag` |
| [ ] | `LinearOperatorFullMatrix` |
| [ ] | `LinearOperatorHouseholder` |
| [ ] | `LinearOperatorIdentity` |
| [ ] | `LinearOperatorInversion` |
| [ ] | `LinearOperatorKronecker` |
| [ ] | `LinearOperatorLowRankUpdate` |
| [ ] | `LinearOperatorLowerTriangular` |
| [ ] | `LinearOperatorPermutation` |
| [ ] | `LinearOperatorScaledIdentity` |
| [ ] | `LinearOperatorToeplitz` |
| [ ] | `LinearOperatorTridiag` |
| [ ] | `LinearOperatorZeros` |
| [ ] | `LionConfig` |
| [ ] | `Load` |
| [x] | `Log` |
| [x] | `Log10` |
| [x] | `Log1P` |
| [x] | `Log2` |
| [ ] | `LogSigmoid` |
| [ ] | `LogSoftmax` |
| [x] | `Logaddexp` |
| [x] | `Logaddexp2` |
| [x] | `Logcumsumexp` |
| [x] | `Loggamma` |
| [x] | `LogicalAnd` |
| [x] | `LogicalNot` |
| [x] | `LogicalOr` |
| [x] | `LogicalXor` |
| [x] | `Logistic` |
| [x] | `Logit` |
| [x] | `Lognormal` |
| [x] | `Logsumexp` |
| [x] | `Lookup` |
| [ ] | `Lstsq` |
| [x] | `Lu` |
| [x] | `LuFactor` |
| [ ] | `LuMatrixInverse` |
| [x] | `LuPivotsToPermutation` |
| [ ] | `LuReconstruct` |
| [x] | `LuSolve` |
| [ ] | `MapFlatValues` |
| [x] | `Matmul` |
| [x] | `MatrixExponential` |
| [x] | `MatrixNorm` |
| [x] | `MatrixPower` |
| [x] | `MatrixRank` |
| [x] | `MatrixTranspose` |
| [x] | `Max` |
| [ ] | `MaxPoolWithIndices` |
| [ ] | `MaxPoolWithIndices_Indices` |
| [ ] | `MaxUnpool1D` |
| [ ] | `MaxUnpool2D` |
| [ ] | `MaxUnpool3D` |
| [x] | `Maximum` |
| [x] | `Maxwell` |
| [x] | `Mean` |
| [x] | `Min` |
| [x] | `Minimum` |
| [x] | `Mod` |
| [ ] | `ModifiedBesselI0` |
| [ ] | `ModifiedBesselI1` |
| [ ] | `ModifiedBesselK0` |
| [ ] | `ModifiedBesselK1` |
| [x] | `Moveaxis` |
| [x] | `MultiDot` |
| [x] | `Multiply` |
| [x] | `MultiplyNoNan` |
| [x] | `MultivariateNormal` |
| [x] | `Mvlgamma` |
| [x] | `NanToNum` |
| [x] | `Nanargmax` |
| [x] | `Nanargmin` |
| [x] | `Nancumprod` |
| [x] | `Nancumsum` |
| [x] | `Nanmax` |
| [x] | `Nanmean` |
| [x] | `Nanmedian` |
| [x] | `Nanmin` |
| [x] | `Nanpercentile` |
| [x] | `Nanprod` |
| [x] | `Nanquantile` |
| [x] | `Nanstd` |
| [x] | `Nansum` |
| [x] | `Nanvar` |
| [x] | `Ndtri` |
| [x] | `Negative` |
| [x] | `Nextafter` |
| [ ] | `Nms` |
| [ ] | `NonblockingLoad` |
| [ ] | `NonblockingSave` |
| [ ] | `Nonzero` |
| [x] | `Norm` |
| [ ] | `NormCdf` |
| [ ] | `NormPdf` |
| [ ] | `Normalization` |
| [x] | `NotEqual` |
| [x] | `Orthogonal` |
| [x] | `Outer` |
| [x] | `Packbits` |
| [ ] | `Pad` |
| [x] | `Pareto` |
| [ ] | `ParseExample` |
| [ ] | `ParseSequenceExample` |
| [ ] | `ParseTensor` |
| [x] | `Partition` |
| [x] | `Pdot` |
| [ ] | `Percentile` |
| [x] | `Permute` |
| [x] | `PerspectiveTransform` |
| [x] | `Pinv` |
| [x] | `Pmean` |
| [x] | `Poisson` |
| [ ] | `PoissonCdf` |
| [ ] | `PoissonPmf` |
| [x] | `Polar` |
| [x] | `Poly` |
| [x] | `Polyadd` |
| [x] | `Polyder` |
| [x] | `Polydiv` |
| [x] | `Polyfit` |
| [x] | `Polygamma` |
| [x] | `Polyint` |
| [x] | `Polymul` |
| [x] | `Polysub` |
| [x] | `Polyval` |
| [x] | `PopulationCount` |
| [x] | `Positive` |
| [x] | `Posterize` |
| [x] | `Power` |
| [x] | `PowerIteration` |
| [x] | `Prod` |
| [x] | `Psum` |
| [x] | `PutAlongAxis` |
| [ ] | `Qdwh` |
| [x] | `Qr` |
| [ ] | `Quantile` |
| [x] | `Rad2Deg` |
| [ ] | `Radians` |
| [x] | `RaggedAdd` |
| [x] | `RaggedConstant` |
| [x] | `RaggedCrossHashed` |
| [x] | `RaggedDot` |
| [x] | `RaggedDynamicBroadcast` |
| [x] | `RaggedGather` |
| [x] | `RaggedMatMul` |
| [x] | `RaggedRange` |
| [x] | `RaggedRowSplitsToSegmentIds` |
| [x] | `RaggedSegmentIdsToRowSplits` |
| [x] | `RaggedStack` |
| [x] | `RaggedStackDynamicPartitions` |
| [x] | `RaggedTensorToDense` |
| [x] | `RandomGammaGrad` |
| [x] | `RandomGammaP` |
| [ ] | `Rank` |
| [ ] | `RavelMultiIndex` |
| [x] | `RawConv2D` |
| [x] | `RawMatMul` |
| [x] | `RawMerge` |
| [x] | `RawSwitch` |
| [x] | `Rayleigh` |
| [ ] | `ReadFile` |
| [x] | `ReadVariable` |
| [x] | `Real` |
| [x] | `Reciprocal` |
| [x] | `ReciprocalNoNan` |
| [x] | `ReduceEuclideanNorm` |
| [x] | `ReducePrecision` |
| [x] | `RegexFullMatch` |
| [x] | `RegexReplace` |
| [ ] | `Rem` |
| [x] | `Remainder` |
| [x] | `Repeat` |
| [x] | `Reshape` |
| [x] | `Resize` |
| [x] | `Rfft` |
| [ ] | `Rfft2` |
| [x] | `Rfftfreq` |
| [ ] | `Rfftn` |
| [x] | `Rfftnd` |
| [x] | `RgbToGrayscale` |
| [x] | `RightShift` |
| [ ] | `Rint` |
| [x] | `RngBitGenerator` |
| [x] | `RngUniform` |
| [x] | `Roll` |
| [x] | `Roots` |
| [x] | `Round` |
| [x] | `RowStack` |
| [x] | `Rsqrt` |
| [ ] | `Save` |
| [ ] | `SaveGguf` |
| [ ] | `Savez` |
| [ ] | `SavezCompressed` |
| [ ] | `ScalarMul` |
| [ ] | `ScaledDotProductAttention` |
| [x] | `Scan` |
| [x] | `Scatter` |
| [x] | `ScatterAdd` |
| [x] | `ScatterApply` |
| [x] | `ScatterMax` |
| [x] | `ScatterMin` |
| [x] | `ScatterMul` |
| [x] | `ScatterNd` |
| [x] | `Schur` |
| [x] | `SearchSorted` |
| [ ] | `Searchsorted` |
| [x] | `SegmentedMm` |
| [x] | `Select` |
| [ ] | `SerializeTensor` |
| [x] | `Setdiff1d` |
| [x] | `Setxor1d` |
| [ ] | `ShiftedChebyshevPolynomialT` |
| [ ] | `ShiftedChebyshevPolynomialU` |
| [ ] | `ShiftedChebyshevPolynomialV` |
| [ ] | `ShiftedChebyshevPolynomialW` |
| [x] | `Sign` |
| [x] | `Signbit` |
| [x] | `Sin` |
| [x] | `Sinc` |
| [x] | `Sinh` |
| [ ] | `Size` |
| [x] | `Slice` |
| [x] | `SliceInDim` |
| [x] | `Slogdet` |
| [ ] | `Smm` |
| [x] | `SobolSample` |
| [ ] | `Softsign` |
| [x] | `Solarize` |
| [x] | `Solve` |
| [ ] | `SolveEx` |
| [x] | `Sort` |
| [ ] | `SortComplex` |
| [x] | `SortKeyVal` |
| [ ] | `SpaceToBatch` |
| [ ] | `SpaceToBatchND` |
| [x] | `SparseAdd` |
| [x] | `SparseBincount` |
| [ ] | `SparseConcat` |
| [x] | `SparseCrossHashed` |
| [x] | `SparseDenseMatMul` |
| [x] | `SparseExpandDims` |
| [x] | `SparseEye` |
| [x] | `SparseFillEmptyRows` |
| [x] | `SparseMapValues` |
| [x] | `SparseMask` |
| [x] | `SparseMaximum` |
| [x] | `SparseMinimum` |
| [ ] | `SparsePlus` |
| [x] | `SparseReduceMax` |
| [x] | `SparseReduceSum` |
| [x] | `SparseReorder` |
| [x] | `SparseResetShape` |
| [x] | `SparseReshape` |
| [x] | `SparseRetain` |
| [ ] | `SparseSampledAdd` |
| [x] | `SparseSegmentMean` |
| [x] | `SparseSegmentSqrtN` |
| [x] | `SparseSegmentSum` |
| [ ] | `SparseSigmoid` |
| [x] | `SparseSlice` |
| [x] | `SparseSoftmax` |
| [ ] | `SparseSplit` |
| [ ] | `SparseToDense` |
| [x] | `SparseToIndicator` |
| [x] | `SparseTranspose` |
| [x] | `SpecialGamma` |
| [x] | `Spence` |
| [x] | `Split` |
| [x] | `Sqrt` |
| [x] | `Sqrtm` |
| [x] | `Square` |
| [x] | `SquaredDifference` |
| [x] | `Squeeze` |
| [x] | `Stack` |
| [x] | `Std` |
| [x] | `Stft` |
| [x] | `StridedSlice` |
| [x] | `StringJoin` |
| [x] | `StringLength` |
| [x] | `StringLookup` |
| [x] | `StringLower` |
| [x] | `StringSplit` |
| [x] | `StringSubstr` |
| [x] | `StringToHash` |
| [x] | `StringToNumber` |
| [x] | `StringUpper` |
| [x] | `Subtract` |
| [x] | `Sum` |
| [x] | `Svd` |
| [x] | `Svdvals` |
| [x] | `Swapaxes` |
| [x] | `Switch` |
| [x] | `T` |
| [x] | `Take` |
| [x] | `TakeAlongAxis` |
| [x] | `Tan` |
| [x] | `Tanh` |
| [x] | `TensorArrayRead` |
| [x] | `TensorArrayStack` |
| [x] | `TensorArrayWrite` |
| [x] | `TensorScatterSub` |
| [x] | `TensorScatterUpdate` |
| [x] | `Tensordot` |
| [x] | `Tensorinv` |
| [x] | `Tensorsolve` |
| [x] | `TextVectorization` |
| [x] | `Tile` |
| [x] | `TopK` |
| [x] | `Trace` |
| [x] | `Transpose` |
| [x] | `TrapezoidalIntegral` |
| [x] | `TriInv` |
| [x] | `Triangular` |
| [x] | `TriangularSolve` |
| [x] | `Tridiagonal` |
| [ ] | `TridiagonalMatmul` |
| [x] | `TridiagonalSolve` |
| [x] | `TrueDivide` |
| [x] | `Trunc` |
| [x] | `TruncateDiv` |
| [x] | `TruncateMod` |
| [x] | `Unfold` |
| [x] | `Union1d` |
| [ ] | `Unique` |
| [x] | `UniqueAll` |
| [x] | `UniqueCounts` |
| [x] | `UniqueInverse` |
| [x] | `UniqueValues` |
| [x] | `Unpackbits` |
| [x] | `UnravelIndex` |
| [ ] | `Unstack` |
| [ ] | `UpdateSlice` |
| [x] | `Variance` |
| [x] | `Vdot` |
| [x] | `Vecdot` |
| [x] | `VectorNorm` |
| [x] | `Vsplit` |
| [x] | `Vstack` |
| [x] | `Wald` |
| [x] | `WeibullMin` |
| [ ] | `Welch` |
| [x] | `Where` |
| [ ] | `WindowHamming` |
| [ ] | `WindowHann` |
| [x] | `WrapKeyData` |
| [ ] | `WriteFile` |
| [x] | `Xdivy` |
| [x] | `Xlog1py` |
| [x] | `Xlogy` |
| [x] | `ZeroFraction` |
| [x] | `Zeta` |
| [ ] | `binomial` |
| [ ] | `categorical` |
| [ ] | `choice` |
| [ ] | `dirichlet` |
| [ ] | `permutation` |
| [ ] | `truncated_normal` |
