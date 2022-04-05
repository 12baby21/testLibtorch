//
// Created by 16586 on 2022/3/29.
//

#ifndef HARD_API_XBNZ_HWH_H
#define HARD_API_XBNZ_HWH_H
// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2021.1 (64-bit)
// Copyright 1986-2021 Xilinx, Inc. All Rights Reserved.
// ==============================================================
// control
// 0x00 : Control signals
//        bit 0  - ap_start (Read/Write/COH)
//        bit 1  - ap_done (Read/COR)
//        bit 2  - ap_idle (Read)
//        bit 3  - ap_ready (Read/COR)
//        bit 7  - auto_restart (Read/Write)
//        others - reserved
// 0x04 : Global Interrupt Enable Register
//        bit 0  - Global Interrupt Enable (Read/Write)
//        others - reserved
// 0x08 : IP Interrupt Enable Register (Read/Write)
//        bit 0  - enable ap_done interrupt (Read/Write)
//        bit 1  - enable ap_ready interrupt (Read/Write)
//        others - reserved
// 0x0c : IP Interrupt Status Register (Read/TOW)
//        bit 0  - ap_done (COR/TOW)
//        bit 1  - ap_ready (COR/TOW)
//        others - reserved
// 0x10 : Data signal of ap_return
//        bit 31~0 - ap_return[31:0] (Read)
// 0x18 : Data signal of rp
//        bit 31~0 - rp[31:0] (Read/Write)
// 0x1c : reserved
// 0x20 : Data signal of up
//        bit 31~0 - up[31:0] (Read/Write)
// 0x24 : reserved
// 0x28 : Data signal of un
//        bit 31~0 - un[31:0] (Read/Write)
// 0x2c : reserved
// 0x30 : Data signal of vp
//        bit 31~0 - vp[31:0] (Read/Write)
// 0x34 : reserved
// 0x38 : Data signal of vn
//        bit 31~0 - vn[31:0] (Read/Write)
// 0x3c : reserved
// 0x40 : Data signal of op
//        bit 31~0 - op[31:0] (Read/Write)
// 0x44 : reserved
// (SC = Self Clear, COR = Clear on Read, TOW = Toggle on Write, COH = Clear on Handshake)

#define XBNZ_CONTROL_ADDR_AP_CTRL   0x00
#define XBNZ_CONTROL_ADDR_GIE       0x04
#define XBNZ_CONTROL_ADDR_IER       0x08
#define XBNZ_CONTROL_ADDR_ISR       0x0c
#define XBNZ_CONTROL_ADDR_AP_RETURN 0x10
#define XBNZ_CONTROL_BITS_AP_RETURN 32
#define XBNZ_CONTROL_ADDR_RP_DATA   0x18
#define XBNZ_CONTROL_BITS_RP_DATA   32
#define XBNZ_CONTROL_ADDR_UP_DATA   0x20
#define XBNZ_CONTROL_BITS_UP_DATA   32
#define XBNZ_CONTROL_ADDR_UN_DATA   0x28
#define XBNZ_CONTROL_BITS_UN_DATA   32
#define XBNZ_CONTROL_ADDR_VP_DATA   0x30
#define XBNZ_CONTROL_BITS_VP_DATA   32
#define XBNZ_CONTROL_ADDR_VN_DATA   0x38
#define XBNZ_CONTROL_BITS_VN_DATA   32
#define XBNZ_CONTROL_ADDR_OP_DATA   0x40
#define XBNZ_CONTROL_BITS_OP_DATA   32


#endif //HARD_API_XBNZ_HWH_H
