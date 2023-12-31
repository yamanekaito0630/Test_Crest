﻿// Crest Ocean System

// This file is subject to the MIT License as seen in the root of this folder structure (LICENSE)

namespace Crest
{
    using Crest.EditorHelpers;
    using UnityEditor;
    using UnityEngine;

    public class LayerAttribute : DecoratedPropertyAttribute
    {
#if UNITY_EDITOR
        internal override void OnGUI(Rect position, SerializedProperty property, GUIContent label, DecoratedDrawer drawer)
        {
            property.intValue = EditorGUI.LayerField(position, label, property.intValue);
        }
#endif
    }
}
