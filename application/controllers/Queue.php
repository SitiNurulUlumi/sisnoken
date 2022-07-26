<?php

class Queue extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();

        // load model
        $this->load->model('ship_model');
        $this->load->model('queue_model');
        $this->load->model('log_model');
        $this->load->model('user_model');

        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));

        // load library
        $this->load->library(array('session', 'form_validation'));

        // autentifikasi
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') == 1) {
            redirect("auth/logout");
        }
    }

    public function index()
    {
        $this->list;
    }

    public function list()
    {
        // die(var_dump(tgl_indo('2021-09-11 10:01:07')));;
        $data['title'] = 'Data Antrian';
        $data['subtitle'] = 'List kapal dalam proses';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('queue/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('queue/script');
    }

    public function ajax_list()
    {
        $list = $this->queue_model->get_datatables($admin = true);
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $ship) {
            $no++;
            switch ($ship->status) {
                case '0':
                    $txt = 'Selesai';
                    $color = 'green';
                    $btnAction = FALSE;
                    $btnAction2 = FALSE;
                    break;
                case '1':
                    $txt = 'Proses Bongkar';
                    $color = 'blue';
                    $btnAction = '<a class="btn btn-sm btn-success" href="javascript:void()" title="Ubah Selesai" onclick="finish(' . "'" . $ship->id . "'" . ')"><i class="fa fa-check"></i></a>';
                    $btnAction2 = FALSE;
                    break;
                case '2':
                    $txt = 'Dalam Antrian';
                    $color = 'yellow';
                    $btnAction = '<a class="btn btn-sm btn-warning" href="javascript:void()" title="Ubah Bongkar" onclick="process(' . "'" . $ship->id . "'" . ')"><i class="fa fa-anchor"></i></a>';
                    $btnAction2 = '<a class="btn btn-sm btn-danger" href="javascript:void()" title="Batalkan" onclick="cancel(' . "'" . $ship->id . "'" . ')"><i class="fa fa-times"></i></a>';
                    break;
                case '3':
                    $txt = 'Pending';
                    $color = 'grey';
                    $btnAction = '<a class="btn btn-sm btn-warning" href="javascript:void()" title="Ubah Antrian" onclick="queue(' . "'" . $ship->id . "'" . ')"><i class="fa fa-list"></i></a>';
                    $btnAction2 = '<a class="btn btn-sm btn-danger" href="javascript:void()" title="Batalkan" onclick="cancel(' . "'" . $ship->id . "'" . ')"><i class="fa fa-times"></i></a>';
                    break;
                case '4':
                    $txt = 'Dibatalkan';
                    $color = 'red';
                    $btnAction = FALSE;
                    $btnAction2 = FALSE;
                    break;
            }
            if ($txt == 'Selesai') {
                $estimate = '';
            } else {
                $estimate = strtotime($ship->estimate);
            }
            $awal  = date_create(); // waktu sekarang
            $akhir = date_create($ship->estimate); //waktu estimasi
            $diff  = date_diff($awal, $akhir); //cari selisih
            $hari = $diff->d . ' Hari, ';
            $jam = $diff->h . ' Jam';
            $sign = substr($diff->format("%R%a."), 0, 1); //ambil tanda min atau plus
            // die(var_dump($sign));
            $perkiraan = '';
            if ($estimate == 0) {
                $estimate = '-';
            } else {
                $estimate = tgl_indo($ship->estimate);
                if ($sign !== '-') { //jika tandanya tidak minus
                    $perkiraan = '(' . $hari . $jam . ' Lagi)'; //keluarkan perkiraan
                }
            }
            $row = array();
            $row[] = $no;
            $row[] = $ship->selar;
            $row[] = $ship->ship_name;
            $row[] = tgl_indo($ship->created_at);
            $row[] = $estimate . $perkiraan;
            $row[] = '<span class="badge bg-' . $color . '">' . $txt . '</span>';
            $row[] = '
            <a class="btn btn-sm btn-info" href="javascript:void()" title="Detail Data" onclick="detail(' . "'" . $ship->id . "'" . ')"><i class="fa fa-eye"></i></a>
            ' . $btnAction . ' ' .
                $btnAction2;
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->queue_model->count_all($admin = true),
            "recordsFiltered" => $this->queue_model->count_filtered($admin = true),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }

    // public function ajax_queue($id)
    // {
    //     // cari data id di database
    //     $getData_id = $this->queue_model->get_by_id($id);

    //     // ambil user id dari session dan dari database
    //     $user_id = $this->session->userdata('user_id');
    //     $ship_name = $getData_id->ship_name;
    //     $owner_name = $getData_id->owner_name;
    //     $selar = $getData_id->selar;

    //     // buat txt pesan
    //     $txt = "Memasukan kapal kedalam antrian " . $ship_name . " (" . $selar . "), milik : " . $owner_name;

    //     // masukan dalam array
    //     $data_log = array(
    //         'alert' => 'danger',
    //         'message' => $txt,
    //         'user_id' => $user_id,
    //         'created_at' => timezone()
    //     );

    //     // simpan log
    //     $this->log_model->save($data_log);

    //     $data = array(
    //         'status' => '2',
    //     );

    //     $this->queue_model->editStatus(array('id' => $id), $data);

    //     echo json_encode(array("status" => TRUE));
    // }

    public function ajax_process()
    {
        $this->_validate('estimate');
        $id = $this->input->post('id');
        // cari data id di database
        $getData_id = $this->queue_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $ship_name = $getData_id->ship_name;
        $owner_name = $getData_id->owner_name;
        $selar = $getData_id->selar;

        // buat txt pesan
        $txt = "Memasukan kapal kedalam antrian bongkar " . $ship_name . " (" . $selar . "), milik : " . $owner_name;

        // masukan dalam array
        $data_log = array(
            'alert' => 'warning',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $data = array(
            'estimate' => timezone($this->input->post('estimate')),
            'status' => '1',
        );

        $this->queue_model->editStatus(array('id' => $id), $data);

        echo json_encode(array("status" => TRUE));
    }

    public function ajax_finish($id)
    {
        // cari data id di database
        $getData_id = $this->queue_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $ship_name = $getData_id->ship_name;
        $owner_name = $getData_id->owner_name;
        $selar = $getData_id->selar;

        // buat txt pesan
        $txt = "Menyelesaikan bongkar " . $ship_name . " (" . $selar . "), milik : " . $owner_name;

        // masukan dalam array
        $data_log = array(
            'alert' => 'warning',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $data = array(
            'status' => '0',
        );

        // ubah status queue
        $this->queue_model->editStatus(array('id' => $id), $data);

        $data2 = array(
            'status' => '1',
            'updated_at' => timezone(),
        );

        // ubah status ship
        $this->ship_model->editStatus(array('id_ship' => $getData_id->ship_id), $data2);

        echo json_encode(array("status" => TRUE));
    }

    public function ajax_cancel($id)
    {
        // cari data id di database
        $getData_id = $this->queue_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $ship_name = $getData_id->ship_name;
        $owner_name = $getData_id->owner_name;
        $selar = $getData_id->selar;

        // buat txt pesan
        $txt = "Membatalkan antrian kapal " . $ship_name . " (" . $selar . "), milik : " . $owner_name;

        // masukan dalam array
        $data_log = array(
            'alert' => 'danger',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $data = array(
            'status' => '4',
        );

        // ubah status queue
        $this->queue_model->editStatus(array('id' => $id), $data);

        $data2 = array(
            'status' => '1',
            'updated_at' => timezone(),
        );

        // ubah status ship
        $this->ship_model->editStatus(array('id_ship' => $getData_id->ship_id), $data2);

        echo json_encode(array("status" => TRUE));
    }

    public function add()
    {
        $data['title'] = 'Tambahkan Antrian';
        $data['subtitle'] = 'Tambahkan antrian kapal anda';
        $data['listShip'] = $this->queue_model->get_rows($this->session->userdata('user_id'));
        $data['listPending'] = $this->queue_model->getQueue($this->session->userdata('user_id'), '2');

        // $checkQueue =  $this->queue_model->checkQueue($this->input->post('ship'));

        // if ($checkQueue == false) {
        //     $data['isPending'] = FALSE;
        // } else {
        //     $data['isPending'] = TRUE;
        // }

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('queue/queue', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('queue/script', $data);
    }

    public function ajax_add()
    {
        $this->_validate('add');

        $getShip =  $this->ship_model->get_by_id($this->input->post('ship'), $this->session->userdata('user_id'));
        if ($getShip) {
            $data = array(
                'ship_id' => $getShip->id_ship,
                'owner_id' => $this->session->userdata('user_id'),
                'selar' => $getShip->selar,
                'ship_name' => $getShip->ship_name,
                'owner_name' => $getShip->fullname,
                'captain' => $getShip->captain,
                'estimate' => date('Y-m-d H:i:s', 0000000000),
                'status' => '2',
                'created_at' => timezone(),
            );

            $data2 = array(
                'status' => '2',
                'updated_at' => timezone(),
            );

            $this->ship_model->editStatus(array('id_ship' => $getShip->id_ship), $data2);
            $this->queue_model->save($data);
            echo json_encode(array("status" => TRUE));
        } else {
            die();
        }
    }

    private function _validate($string = false)
    {
        $data = array();
        $data['error_string'] = array();
        $data['inputerror'] = array();
        $data['status'] = TRUE;
        if ($string == 'estimate') {
            if ($this->input->post('id') == '') {
                $data['inputerror'][] = 'id';
                $data['error_string'][] = 'Error ID';
                $data['status'] = FALSE;
            }
            if ($this->input->post('estimate') == '') {
                $data['inputerror'][] = 'estimate';
                $data['error_string'][] = 'Estimasi wajib diisi';
                $data['status'] = FALSE;
            }
        } else {

            //   validasi ship
            if ($this->input->post('ship') == '') {
                $data['inputerror'][] = 'ship';
                $data['error_string'][] = 'Kapal wajib diisi';
                $data['status'] = FALSE;
            }
        }


        if ($data['status'] === FALSE) {
            echo json_encode($data);
            exit();
        }
    }

    public function ajax_read_info()
    {
        $this->session->set_userdata('is_info_read', true);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_cek($id)
    {
        $data = $this->queue_model->get_by_id($id);
        echo json_encode($data);
    }
}
