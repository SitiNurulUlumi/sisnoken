<?php

class Users extends CI_Controller
{

    public function __construct()
    {
        parent::__construct();

        // load model
        $this->load->model('user_model');
        $this->load->model('log_model');

        // load helper
        $this->load->helper(array('form', 'url', 'timezone_helper'));

        // load library
        $this->load->library(array('session', 'form_validation'));

        // autentifikasi
        if (!$this->session->userdata('is_logged_in')) {
            redirect("home");
        } elseif ($this->session->userdata('role') == 3) {
            redirect("auth/logout");
        }
    }

    public function index()
    {
        $this->list();
    }

    public function list()
    {
        $data['title'] = 'Data Pemilik Kendaraan';
        $data['subtitle'] = 'List Pemilik Kendaraan';
        $data['role_list'] = '3';

        $this->load->view('templates/admin-header', $data);
        $this->load->view('templates/admin-topbar', $data);
        $this->load->view('users/list', $data);
        $this->load->view('templates/admin-footer');
        $this->load->view('users/script', $data);
    }

    public function ajax_list($role = false)
    {
        $list = $this->user_model->get_datatables($role);
        $data = array();
        $no = $_POST['start'];
        foreach ($list as $user) {
            $no++;
            $row = array();
            if ($this->session->userdata('role') == 2) {
                $btnTambahKendaraan = '<a class="btn btn-sm btn-success" href="javascript:void()" title="Tambah Kendaraan" onclick="addPlat(' . "'" . $user->id . "'" . ')"><i class="fa fa-plus"></i></a>';
            } else {
                $btnTambahKendaraan = false;
            }
            $row[] = '<input type="checkbox" class="data-check" value="' . $user->id . '" onclick="showBottomDelete()"/>';
            $row[] = $btnTambahKendaraan . '
            <a class="btn btn-sm btn-warning" href="javascript:void()" title="Ubah Data" onclick="edit(' . "'" . $user->id . "'" . ')"><i class="glyphicon glyphicon-pencil"></i></a>
            <a class="btn btn-sm btn-warning" href="javascript:void()" title="Reset Password" onclick="resetPassword(' . "'" . $user->id . "'" . ')"><i class="fa fa-key"></i></a>
            <a class="btn btn-sm btn-danger" href="javascript:void()" title="Hapus Data" onclick="deleteData(' . "'" . $user->id . "'" . ')"><i class="glyphicon glyphicon-trash"></i></a>
            ';
            $row[] = $user->fullname;
            $row[] = $user->email;
            $data[] = $row;
        }
        $output = array(
            "draw" => $_POST['draw'],
            "recordsTotal" => $this->user_model->count_all($role),
            "recordsFiltered" => $this->user_model->count_filtered($role),
            "data" => $data,
        );
        //output to json format
        echo json_encode($output);
    }

    public function ajax_edit($id)
    {
        $data = $this->user_model->get_by_id($id);
        echo json_encode($data);
    }

    public function ajax_add($role = false)
    {
        $this->_validate('add');
        $data = array(
            'fullname' => $this->input->post('fullname'),
            'email' => $this->input->post('email'),
            'password' => md5($this->input->post('repassword')),
            'role' => $role,
            'updated_at' => timezone()
        );
        $insert = $this->user_model->save($data);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_update()
    {
        $this->_validate();
        $data = array(
            'fullname' => $this->input->post('fullname'),
            'email' => $this->input->post('email'),
            'updated_at' => timezone()
        );
        $this->user_model->update(array('id' => $this->input->post('id')), $data);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_delete($id)
    {
        // cari data id di database
        $getData_id = $this->user_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $name = $getData_id->fullname;
        $email = $getData_id->email;

        // buat txt pesan
        $txt = "Menghapus data " . $name . " (" . $email . ")";

        // masukan dalam array
        $data_log = array(
            'alert' => 'danger',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $this->user_model->delete_by_id($id);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_password($id)
    {
        // cari data id di database
        $getData_id = $this->user_model->get_by_id($id);

        // ambil user id dari session dan dari database
        $user_id = $this->session->userdata('user_id');
        $name = $getData_id->fullname;
        $email = $getData_id->email;

        // buat txt pesan
        $txt = "Merubah password " . $name . " (" . $email . ")";

        // masukan dalam array
        $data_log = array(
            'alert' => 'warning',
            'message' => $txt,
            'user_id' => $user_id,
            'created_at' => timezone()
        );

        // simpan log
        $this->log_model->save($data_log);

        $password = md5('1234');
        $data = array(
            'password' => $password,
        );
        $this->user_model->update(array('id' => $id), $data);
        echo json_encode(array("status" => TRUE));
    }

    public function ajax_list_delete()
    {
        $list_id = $this->input->post('id');
        foreach ($list_id as $id) {
            // cari data id di database
            $getData_id = $this->user_model->get_by_id($id);

            // ambil user id dari session dan dari database
            $user_id = $this->session->userdata('user_id');
            $name = $getData_id->fullname;
            $email = $getData_id->email;

            // buat txt pesan
            $txt = "Menghapus data " . $name . " (" . $email . ")";

            // masukan dalam array
            $data_log = array(
                'alert' => 'danger',
                'message' => $txt,
                'user_id' => $user_id,
                'created_at' => timezone()
            );

            // simpan log
            $this->log_model->save($data_log);

            $this->user_model->delete_by_id($id);
        }
        echo json_encode(array("status" => TRUE));
    }

    private function _validate($string = false)
    {
        $data = array();
        $data['error_string'] = array();
        $data['inputerror'] = array();
        $data['status'] = TRUE;

        //   validasi fullname
        if ($this->input->post('fullname') == '') {
            $this->session->set_flashdata('msg_error', 'Login credentials does not match!');
            $data['inputerror'][] = 'fullname';
            $data['error_string'][] = 'First name wajib diisi';
            $data['status'] = FALSE;
        } elseif (!$this->_validate_string($this->input->post('fullname'))) {
            $data['inputerror'][] = 'fullname';
            $data['error_string'][] = 'Hanya boleh input karakter';
            $data['status'] = FALSE;
        }


        //   validasi email/username
        if ($this->input->post('email') == '') {
            $data['inputerror'][] = 'email';
            $data['error_string'][] = 'Username wajib diisi';
            $data['status'] = FALSE;
        }

        if ($string == 'add') {
            //   validasi password
            if ($this->input->post('password') == '') {
                $data['inputerror'][] = 'password';
                $data['error_string'][] = 'Password wajib diisi';
                $data['status'] = FALSE;
            }

            //   validasi repassword
            if ($this->input->post('repassword') == '') {
                $data['inputerror'][] = 'repassword';
                $data['error_string'][] = 'Harap ketik ulang password';
                $data['status'] = FALSE;
            } elseif (!$this->_validate_password($this->input->post('password'), $this->input->post('repassword'))) {
                $data['inputerror'][] = 'repassword';
                $data['error_string'][] = 'Password tidak sama, silakan periksa ulang';
                $data['status'] = FALSE;
            }
        }

        if ($data['status'] === FALSE) {
            echo json_encode($data);
            exit();
        }
    }

    private function _validate_string($string)
    {
        $allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ";
        for ($i = 0; $i < strlen($string); $i++) {
            if (strpos($allowed, substr($string, $i, 1)) === FALSE) {
                return FALSE;
            }
        }

        return TRUE;
    }

    private function _validate_password($string1, $string2)
    {
        if ($string1 != $string2) {
            return FALSE;
        }

        return TRUE;
    }
}
