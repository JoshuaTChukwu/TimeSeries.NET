#if NETSTANDARD2_0

// ReSharper disable once CheckNamespace
namespace System.Runtime.CompilerServices;

/// <summary>
/// Compiler shim enabling <c>init</c> accessors and records on netstandard2.0.
/// Present only in the netstandard2.0 build; never part of the public surface.
/// </summary>
[System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
internal static class IsExternalInit
{
}

#endif
