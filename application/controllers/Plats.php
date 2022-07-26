<?php

class Plats extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();

        // load model
        $this->load->model('plat_model');
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
        $this->list();
    }

    public function list()
    {
        $data['title'] = 'Data Kendaraan';
        $data['subtitle'] = 'List Kendaraan';
        if ($this->session->userdata('role') == 3) {
            $sessId = $this->session->userdata('user_id');
            $data['sessId'] = $sessId;
        } else {
            $data['sessId'] = false;
        }

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('ships/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('ships/script', $data);
    }


    public function ajax_add()
    {
        $this->_validate('add');
        if ($this->session->userdata('role') == 3) {
            $idUser = $this->session->userdata('user_id');
        } else {
            $idUser = $this->input->post('id');
        }
        $data = array(
            'plat_nomor' => $this->input->post('plat_nomor'),
            'owner_id' => $idUser,
            'updated_at' => timezone()
        );
        // die(var_dump($data));
        $insert = $this->plat_model->save($data);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_list()
    {
        if ($this->session->userdata('role') == 3) {
            $sessId = $this->session->userdata('user_id');
        } else {
            $sessId = false;
        }
        $list = $this->plat_model->get_datatables($sessId);
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $plat) {
            if ($this->session->userdata('role') == 3) {
                $btnDetail = false;
                $row3 = $plat->captain;
                $no++;
                $row = array();
                $row[] = $no;
                $row[] = $plat->plat_nomor;
                $row[] = $row3;
            } else {
                $btnDetail = '<a class="btn btn-sm btn-info" href="javascript:void()" title="Detail Data" onclick="detail(' . "'" . $plat->id_plat . "'" . ')"><i class="fa fa-eye"></i></a>';
                $btnAntrian = false;
                $row3  = $plat->fullname;
                $no++;
                $row = array();
                $row[] = '<input type="checkbox" class="data-check" value="' . $plat->id_plat . '" onclick="showBottomDelete()"/>';
                $row[] = $btnDetail . '

            <a class="btn btn-sm btn-warning" href="javascript:void()" title="Ubah Data" onclick="edit(' . "'" . $plat->id_plat . "'" . ')"><i class="glyphicon glyphicon-pencil"></i></a>
            <a class="btn btn-sm btn-danger" href="javascript:void()" title="Hapus Data" onclick="deleteData(' . "'" . $plat->id_plat . "'" . ')"><i class="glyphicon glyphicon-trash"></i></a>
            ';
                $row[] = $plat->plat_nomor;
                $row[] = $row3;
            }

            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->plat_model->count_all($sessId),
            "recordsFiltered" => $this->plat_model->count_filtered($sessId),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }

    public function ajax_edit($id)
    {
        $data = $this->plat_model->get_by_id($id);
        echo json_encode($data);
    }

    public function add($id = false)
    {
        $dataUser = $this->user_model->get_by_id($id);
        // $fullname = $dataUser->fullname;
        // die(var_dump($data->fullname));

        $data['title'] = 'Tambah Kapal';
        $data['subtitle'] = 'Menambahkan kapal untuk ' . $dataUser->fullname;
        $data['idUser'] = $id;

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('plats/add', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('plats/script', $data);
    }

    public function ajax_update()
    {
        $this->_validate();
        // cari data id di database
        $getData_id = $this->plat_model->get_by_id($this->input->post('id'));

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $fullname = $getData_id->fullname;
        $plat_nomor = $getData_id->plat_nomor;

        // buat txt pesan
        $txt = "Merubah nomor kendaraan dengan plat (" . $plat_nomor . "), milik : " . $fullname;

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
            'plat_nomor' => $this->input->post('plat_nomor'),
            'updated_at' => timezone()
        );
        $this->plat_model->update(array('id_plat' => $this->input->post('id')), $data);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_delete($id)
    {
        // cari data id di database
        $getData_id = $this->plat_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $fullname = $getData_id->fullname;
        $plat_nomor = $getData_id->plat_nomor;

        // buat txt pesan
        $txt = "Menghapus nomor kendaraan dengan plat (" . $plat_nomor . "), milik : " . $fullname;

        // masukan dalam array
        $data_log = array(
            'alert' => 'danger',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $this->plat_model->delete_by_id($id);
        echo json_encode(array("status" => TRUE));
    }


    public function ajax_list_delete()
    {
        $list_id = $this->input->post('id');
        foreach ($list_id as $id) {
            // cari data id di database
            $getData_id = $this->plat_model->get_by_id($id);

            // ambil user id dari session dan dari database
            $user_id = $this->session->userdata('user_id');
            $fullname = $getData_id->fullname;
            $plat_nomor = $getData_id->plat_nomor;

            // buat txt pesan
            $txt = "Menghapus nomor kendaraan dengan plat (" . $plat_nomor . "), milik : " . $fullname;

            // masukan dalam array
            $data_log = array(
                'alert' => 'danger',
                'message' => $txt,
                'user_id' => $user_id,
                'created_at' => timezone()
            );

            // simpan log
            $this->log_model->save($data_log);

            $this->plat_model->delete_by_id($id);
        }
        echo json_encode(array("status" => TRUE));
    }

    private function _validate($string = false)
    {
        $data = array();
        $data['error_string'] = array();
        $data['inputerror'] = array();
        $data['status'] = TRUE;

        //   validasi plat_nomor
        if ($this->input->post('plat_nomor') == '') {
            $data['inputerror'][] = 'plat_nomor';
            $data['error_string'][] = 'Nomor Kendaraan wajib diisi';
            $data['status'] = FALSE;
        } elseif ($this->plat_model->get_by_plat_nomor($this->input->post('plat_nomor'))) {
            $data['inputerror'][] = 'plat_nomor';
            $data['error_string'][] = 'Nomor Kendaraan sudah ada';
            $data['status'] = FALSE;
        }

        if ($data['status'] === FALSE) {
            echo json_encode($data);
            exit();
        }
    }
}
