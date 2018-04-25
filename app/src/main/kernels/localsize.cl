enum {
    idtype_globalsize_x = 1,
    idtype_globalsize_y = 2,
    idtype_globalsize_z = 3,

    idtype_localsize_x  = 4,
    idtype_localsize_y  = 5,
    idtype_localsize_z  = 6,

    idtype_globalid_x   = 7,
    idtype_globalid_y   = 8,
    idtype_globalid_z   = 9,

    idtype_groupid_x    = 10,
    idtype_groupid_y    = 11,
    idtype_groupid_z    = 12,

    idtype_localid_x    = 13,
    idtype_localid_y    = 14,
    idtype_localid_z    = 15
};

__kernel void ReadLocalSize(__global int* outIds, int inWidth, int inHeight, int inPitch, int inIdType)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < inWidth && y < inHeight)
    {
        int index = x + inPitch*y;

        int data = -1;
        switch (inIdType)
        {
        case idtype_globalsize_x:
        case idtype_globalsize_y:
        case idtype_globalsize_z:
            data = get_global_size(inIdType - idtype_globalsize_x);
            break;

        case idtype_localsize_x:
        case idtype_localsize_y:
        case idtype_localsize_z:
            data = get_local_size(inIdType - idtype_localsize_x);
            break;

        case idtype_globalid_x:
        case idtype_globalid_y:
        case idtype_globalid_z:
            data = get_global_id(inIdType - idtype_globalid_x);
            break;

        case idtype_groupid_x:
        case idtype_groupid_y:
        case idtype_groupid_z:
            data = get_group_id(inIdType - idtype_groupid_x);
            break;

        case idtype_localid_x:
        case idtype_localid_y:
        case idtype_localid_z:
            data = get_local_id(inIdType - idtype_localid_x);
            break;
        }

        outIds[index] = data;
    }
}
