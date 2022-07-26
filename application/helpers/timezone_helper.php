<?php if (!defined('BASEPATH')) exit('No direct script access allowed');

if (!function_exists('timezone')) {
    function timezone($id = false)
    {
        if ($id) {
            $string = date('Y-m-d H:i:s', strtotime($id . ' days'));
        } else {
            $string = date('Y-m-d H:i:s');
        }
        date_default_timezone_set('Asia/Jakarta'); # add your city to set local time zone
        return $string;
    }

    function tgl_indo($tanggal)
    {
        $tanggal = strtotime($tanggal);
        $tanggal = date('Y-m-d', $tanggal);
        // echo '<pre>';
        // die(var_dump($tanggal));
        $bulan = array(
            1 =>   'Januari',
            'Februari',
            'Maret',
            'April',
            'Mei',
            'Juni',
            'Juli',
            'Agustus',
            'September',
            'Oktober',
            'November',
            'Desember'
        );
        $pecahkan = explode('-', $tanggal);

        // variabel pecahkan 0 = tanggal
        // variabel pecahkan 1 = bulan
        // variabel pecahkan 2 = tahun

        return $pecahkan[2] . ' ' . $bulan[(int)$pecahkan[1]] . ' ' . $pecahkan[0] . ' ' . date('h:i:s');
    }
}
